use proc_macro::TokenStream;
use quote::ToTokens;
use quote::format_ident;
use quote::quote;
use stringcase::Caser;
use syn::{FnArg, Ident, Pat, Token, parse_macro_input, punctuated::Punctuated};

/// A field that the interpreter generates a set of opcodes for: an entry
/// `name = Type` of `#[interpreter(fields(...))]`, optionally preceded by
/// `#[cfg(...)]`
///
/// `name` is the prefix of the field's opcode names (`bn254` gives
/// `bn254_add_field` and `Bn254AddField`), and its PascalCase form is the
/// field's `FieldId` variant. A `cfg`-gated field is a test instance: it has no
/// `FieldId` variant, so `FieldOpCode::for_field` cannot select it, and it must
/// be the last field so that its gating moves no other opcode's number.
struct FieldInstance {
    name: Ident,
    ty: syn::Type,
    cfg: Option<proc_macro2::TokenStream>,
}

impl syn::parse::Parse for FieldInstance {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let attrs = input.call(syn::Attribute::parse_outer)?;
        let cfg = match attrs.as_slice() {
            [] => None,
            [attr] if attr.path().is_ident("cfg") => Some(attr.meta.require_list()?.tokens.clone()),
            _ => {
                return Err(input.error("a field entry takes at most one attribute, `#[cfg(..)]`"));
            }
        };
        let name: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let ty: syn::Type = input.parse()?;
        Ok(FieldInstance { name, ty, cfg })
    }
}

/// The `#[interpreter(fields(..))]` argument.
struct InterpreterArgs {
    fields: Vec<FieldInstance>,
}

impl syn::parse::Parse for InterpreterArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let key: Ident = input.parse()?;
        if key != "fields" {
            return Err(syn::Error::new(key.span(), "expected `fields(..)`"));
        }
        let content;
        syn::parenthesized!(content in input);
        let fields = Punctuated::<FieldInstance, Token![,]>::parse_terminated(&content)?;
        Ok(InterpreterArgs {
            fields: fields.into_iter().collect(),
        })
    }
}

/// `#[cfg(..)]` for an item that exists only under `cfg`, or nothing.
fn cfg_attr(cfg: &Option<proc_macro2::TokenStream>) -> proc_macro2::TokenStream {
    match cfg {
        Some(cfg) => quote! { #[cfg(#cfg)] },
        None => quote! {},
    }
}

#[derive(Clone, Debug)]
enum GuestType {
    Field,
    U64,
    Int128,
    Ptr,
    BoxedValue,
}

impl GuestType {
    /// How many frame cells a value of this type covers.
    fn to_cells_tokens(&self, field_ty: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        match self {
            GuestType::Field => quote! { Frame::cells_of::<#field_ty>() },
            GuestType::Int128 => quote! { Frame::cells_of::<Int128>() },
            GuestType::BoxedValue => quote! { Frame::cells_of::<BoxedValue>() },
            GuestType::U64 => quote! { Frame::cells_of::<u64>() },
            // A pointer is read out of one cell whatever it points at, so there is no type here
            // whose size would answer the question.
            GuestType::Ptr => quote! { 1usize },
        }
    }
}

#[derive(Clone, Debug)]
enum HostType {
    U64,
    USize,
    ISize,
    Field,
    JumpTarget,
    Slice(Box<HostType>),
    Tuple(Vec<HostType>),
    FramePosition,
    BoxedLayout,
}

impl HostType {
    fn to_def_tokens(&self, field_ty: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        match self {
            HostType::U64 => quote! { u64 },
            HostType::USize => quote! { usize },
            HostType::ISize => quote! { isize },
            HostType::JumpTarget => quote! { JumpTarget },
            HostType::Field => quote! { #field_ty },
            HostType::Slice(ty) => {
                let child = ty.to_def_tokens(field_ty);
                quote! { Vec<#child> }
            }
            HostType::Tuple(ty) => {
                let children = ty
                    .iter()
                    .map(|ty| ty.to_def_tokens(field_ty))
                    .collect::<Vec<_>>();
                quote! { ( #(#children),* ) }
            }
            HostType::FramePosition => quote! { FramePosition },
            HostType::BoxedLayout => quote! { BoxedLayout },
        }
    }

    fn make_getter(
        &self,
        offset: proc_macro2::Ident,
        pc: proc_macro2::Ident,
        field_ty: &proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        match self {
            HostType::JumpTarget => {
                quote! {
                    unsafe {
                        let r = JumpTarget(*#pc.offset(#offset) as isize);
                        #offset += 1;
                        r
                    }
                }
            }
            HostType::FramePosition => {
                quote! {
                    unsafe {
                        let r = FramePosition(*#pc.offset(#offset) as usize);
                        #offset += 1;
                        r
                    }
                }
            }
            HostType::BoxedLayout => {
                quote! {
                    unsafe {
                        let r = BoxedLayout(*#pc.offset(#offset));
                        #offset += 1;
                        r
                    }
                }
            }
            HostType::U64 => {
                quote! {
                    unsafe {
                        let r = *#pc.offset(#offset);
                        #offset += 1;
                        r
                    }
                }
            }
            HostType::USize => {
                quote! {
                    unsafe {
                        let r = *#pc.offset(#offset) as usize;
                        #offset += 1;
                        r
                    }
                }
            }
            HostType::ISize => {
                quote! {
                    unsafe {
                        let r = *#pc.offset(#offset) as isize;
                        #offset += 1;
                        r
                    }
                }
            }
            HostType::Slice(ty) => {
                let size = ty.measure_size(field_ty);
                quote! {
                    unsafe {
                        let len = *#pc.offset(#offset) as usize;
                        #offset += 1;
                        let r = std::slice::from_raw_parts(
                            #pc.offset(#offset) as *const _,
                            len
                        );
                        #offset += len as isize * (#size) as isize;
                        r
                    }
                }
            }
            // A field immediate is encoded as its storage cells, the form frames hold, so the VM
            // never converts one.
            HostType::Field => {
                quote! {
                    unsafe {
                        let cells = std::slice::from_raw_parts(
                            #pc.offset(#offset),
                            <#field_ty as ::mavros_artifacts::MavrosField>::STORAGE_CELLS,
                        );
                        #offset += cells.len() as isize;
                        <#field_ty as ::mavros_artifacts::MavrosField>::from_frame_cells(cells)
                    }
                }
            }
            HostType::Tuple(_ty) => {
                todo!("tuple getter");
            }
        }
    }

    /// The number of instruction words a value of this type is encoded as, as a `usize` expression:
    /// a field's width is its `STORAGE_CELLS`, which only the compiled crate can evaluate.
    fn measure_size(&self, field_ty: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        match self {
            HostType::U64
            | HostType::USize
            | HostType::ISize
            | HostType::JumpTarget
            | HostType::FramePosition
            | HostType::BoxedLayout => quote! { 1usize },
            HostType::Field => {
                quote! { <#field_ty as ::mavros_artifacts::MavrosField>::STORAGE_CELLS }
            }
            HostType::Slice(_) => panic!("slice size is not known at compile time"),
            HostType::Tuple(ty) => {
                let sizes = ty.iter().map(|ty| ty.measure_size(field_ty));
                quote! { (0usize #(+ #sizes)*) }
            }
        }
    }

    /// Whether a value of this type holds a field element anywhere.
    fn contains_field(&self) -> bool {
        match self {
            HostType::Field => true,
            HostType::Slice(ty) => ty.contains_field(),
            HostType::Tuple(tys) => tys.iter().any(HostType::contains_field),
            HostType::U64
            | HostType::USize
            | HostType::ISize
            | HostType::JumpTarget
            | HostType::FramePosition
            | HostType::BoxedLayout => false,
        }
    }

    fn make_serializer(
        &self,
        i: proc_macro2::TokenStream,
        offset: &proc_macro2::Ident,
        binary: &proc_macro2::Ident,
        jumps_to_fix: &proc_macro2::Ident,
        is_ref: bool,
        field_ty: &proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        match self {
            HostType::JumpTarget => {
                quote! {
                    #jumps_to_fix.push((#binary.len(), #offset));
                    #binary.push(#i.0 as u64);
                    #offset -= 1;
                }
            }
            HostType::FramePosition => {
                quote! {
                    #binary.push(#i.0 as u64);
                    #offset -= 1;
                }
            }
            HostType::BoxedLayout => {
                quote! {
                    #binary.push(#i.0);
                    #offset -= 1;
                }
            }
            HostType::USize | HostType::ISize => {
                let i = if is_ref {
                    quote! { *#i }
                } else {
                    quote! { #i }
                };
                quote! {
                    #binary.push(#i as u64);
                    #offset -= 1;
                }
            }
            HostType::U64 => {
                let i = if is_ref {
                    quote! { *#i }
                } else {
                    quote! { #i }
                };
                quote! {
                    #binary.push(#i);
                    #offset -= 1;
                }
            }
            HostType::Field => {
                let i = if is_ref {
                    quote! { *#i }
                } else {
                    quote! { #i }
                };
                quote! {
                    {
                        let mut cells =
                            [0u64; <#field_ty as ::mavros_artifacts::MavrosField>::STORAGE_CELLS];
                        ::mavros_artifacts::MavrosField::to_frame_cells(#i, &mut cells);
                        #binary.extend_from_slice(&cells);
                        #offset -= cells.len() as isize;
                    }
                }
            }
            HostType::Slice(intp) => {
                let ixed = quote! { elem };
                let child_serializer = intp.make_serializer(
                    ixed.clone(),
                    offset,
                    binary,
                    jumps_to_fix,
                    true,
                    field_ty,
                );
                quote! {
                    #binary.push(#i.len() as u64);
                    #offset -= 1;
                    for elem in #i {
                        #child_serializer;
                    }
                }
            }
            HostType::Tuple(intps) => {
                let mut result = proc_macro2::TokenStream::new();
                for (ix, intp) in intps.iter().enumerate() {
                    let ix = syn::Index::from(ix);
                    let ixed = quote! { #i.#ix };
                    result.extend(intp.make_serializer(
                        ixed,
                        offset,
                        binary,
                        jumps_to_fix,
                        false,
                        field_ty,
                    ));
                }
                result
            }
        }
    }

    fn make_measurer(
        &self,
        binary: &proc_macro2::Ident,
        ix: &proc_macro2::Ident,
        field_ty: &proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        match self {
            HostType::U64
            | HostType::USize
            | HostType::ISize
            | HostType::JumpTarget
            | HostType::BoxedLayout
            | HostType::FramePosition => {
                quote! {
                    #ix += 1;
                }
            }
            HostType::Field => {
                let size = self.measure_size(field_ty);
                quote! {
                    #ix += #size;
                }
            }
            HostType::Slice(child) => {
                let child_size = child.measure_size(field_ty);
                quote! {
                    {
                        let len = #binary[#ix] as usize;
                        #ix += 1;
                        #ix += len * (#child_size);
                    };
                }
            }
            HostType::Tuple(_children) => {
                quote! { todo!("tuple measurer") }
            }
        }
    }
}

/// A run of consecutive frame cells at one encoded `FramePosition`, whose length comes from an
/// integer width the same opcode carries. A wide integer's cells are contiguous, so the position
/// alone cannot say how many to read.
#[derive(Clone, Debug)]
struct SliceInput {
    /// The name of the host parameter holding the operand width in bits.
    ///
    /// It must be declared ahead of the slice, because getters consume the instruction stream in
    /// declaration order.
    width: String,

    /// Whether the handler receives `&mut [u64]` rather than `&[u64]`.
    mutable: bool,
}

#[derive(Clone, Debug)]
enum StructInputType {
    Frame(GuestType),
    Out(GuestType),
    Slice(SliceInput),
    Host(HostType),
}

#[derive(Clone, Debug)]
struct StructInput {
    name: String,
    val: StructInputType,
}

impl StructInput {
    fn make_printer(&self, fields_ident: &proc_macro2::Ident) -> proc_macro2::TokenStream {
        let ident = format_ident!("{}", self.name);
        match &self.val {
            StructInputType::Host(HostType::JumpTarget) => quote! {
                #fields_ident = format!("{} {}", #fields_ident, #ident.0);
            },
            StructInputType::Host(HostType::FramePosition) => quote! {
                #fields_ident = format!("{} %{}", #fields_ident, #ident.0);
            },
            StructInputType::Host(HostType::U64) => quote! {
                #fields_ident = format!("{} {}", #fields_ident, #ident);
            },
            StructInputType::Host(HostType::USize) => quote! {
                #fields_ident = format!("{} {}", #fields_ident, #ident);
            },
            StructInputType::Host(HostType::ISize) => quote! {
                #fields_ident = format!("{} {}", #fields_ident, #ident);
            },
            StructInputType::Host(HostType::Field) => quote! {
                #fields_ident = format!("{} {}", #fields_ident, #ident);
            },
            StructInputType::Frame(_) => quote! {
                #fields_ident = format!("{} %{}", #fields_ident, #ident.0);
            },
            StructInputType::Out(_) | StructInputType::Slice(_) => quote! {
                #fields_ident = format!("{} %{}", #fields_ident, #ident.0);
            },
            StructInputType::Host(
                HostType::Slice(_) | HostType::Tuple(_) | HostType::BoxedLayout,
            ) => quote! {
                #fields_ident = format!("{} _", #fields_ident);
            },
        }
    }
}

#[derive(Clone, Debug)]
enum Input {
    Struct(StructInput),
    Frame,
    VM,
}

#[derive(Clone, Debug)]
struct OpCodeDef {
    /// The opcode's name: its mnemonic, and in PascalCase its `OpCode` variant.
    name: String,

    /// The function the handler calls: `name`, or for one field's copy of a field opcode, the
    /// generic function every field shares.
    fn_name: String,

    is_raw: bool,
    inputs: Vec<Input>,

    /// The type a `Field` operand has: `Field` for a plain opcode, the field's type for a copy.
    field_ty: proc_macro2::TokenStream,

    /// Whether `fn_name` is generic over the field, so the handler names `field_ty` explicitly.
    generic: bool,

    /// `Some` for a field opcode: on a copy, its field's index into `InterpreterArgs::fields`; on
    /// the declared template, `Some(0)`, marking it as one.
    field: Option<usize>,

    /// The `cfg` predicate the copy exists under, for a `cfg`-gated field's copies.
    cfg: Option<proc_macro2::TokenStream>,
}

impl OpCodeDef {
    fn struct_args(&self) -> impl Iterator<Item = &StructInput> {
        self.inputs.iter().filter_map(|input| match input {
            Input::Struct(input) => Some(input),
            _ => None,
        })
    }

    fn struct_name(&self) -> String {
        self.name.to_pascal_case()
    }

    fn matcher(&self) -> proc_macro2::TokenStream {
        let struct_args = self
            .struct_args()
            .map(|input| format_ident!("{}", input.name))
            .collect::<Vec<_>>();
        let name = format_ident!("{}", self.struct_name());
        quote! {
            OpCode::#name { #(#struct_args),* }
        }
    }
}

/// Generates the dispatch machinery for the opcodes declared in the annotated module.
///
/// An opcode's number is its index in the generated list: the declared opcodes in declaration
/// order, then, for each field after the first in `fields(..)`, that field's copies of every
/// `#[field_opcode]` in declaration order. A `#[field_opcode]` function is generic over one type
/// parameter, the field, and the first field's copy takes the function's position, so declaring a
/// further field shouldn't cause any opcode to move.
#[proc_macro_attribute]
pub fn interpreter(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as InterpreterArgs);
    check_fields(&args.fields);
    let i = item.clone();
    let input = parse_macro_input!(i as syn::ItemMod);

    let mut result = proc_macro2::TokenStream::new();
    let items = input.content.unwrap().1;

    let mut codes = vec![];

    for item in items {
        match item {
            syn::Item::Fn(mut func) => {
                if !func.attrs.iter().any(|attr| {
                    attr.path().is_ident("opcode") || attr.path().is_ident("raw_opcode")
                }) {
                    result.extend(func.into_token_stream());
                    continue;
                }

                let old_attrs = func.attrs;
                let mut controls_inline = false;
                let mut is_raw = false;
                let mut is_field = false;
                func.attrs = vec![];
                for attr in old_attrs {
                    if attr.path().is_ident("opcode") {
                        continue;
                    }
                    if attr.path().is_ident("raw_opcode") {
                        is_raw = true;
                        continue;
                    }
                    if attr.path().is_ident("field_opcode") {
                        is_field = true;
                        continue;
                    }
                    if attr.path().is_ident("inline") {
                        controls_inline = true;
                    }
                    func.attrs.push(attr);
                }

                let def = parse_signature(is_raw, is_field, &func.sig);
                codes.push(def);

                for inp in func.sig.inputs.iter_mut() {
                    match inp {
                        FnArg::Receiver(_) => {
                            panic!(
                                "receiver arguments are not allowed in the interpreter definition"
                            );
                        }
                        FnArg::Typed(pat) => {
                            pat.attrs = vec![];
                        }
                    }
                }

                if !controls_inline {
                    result.extend(quote! {
                        #[inline(always)]
                    });
                }
                result.extend(func.into_token_stream());
            }
            _ => {
                panic!("only functions are allowed in the interpreter definition");
            }
        }
    }

    let templates = codes;
    let codes = expand_field_opcodes(&templates, &args.fields);

    for (idx, code) in codes.iter().enumerate() {
        result.extend(gen_handler(idx, code));
    }

    result.extend(gen_opcode_enum(&codes));
    result.extend(gen_opcode_helpers(&codes));
    result.extend(gen_field_opcode_enum(&templates, &codes, &args.fields));

    result.into()
}

/// The first field owns the declared positions, so it cannot be `cfg`-gated; a gated field must be
/// last, so that gating it changes only the numbers past every other opcode.
fn check_fields(fields: &[FieldInstance]) {
    assert!(
        !fields.is_empty(),
        "`fields(..)` must name at least one field"
    );
    assert!(
        fields[0].cfg.is_none(),
        "the first field takes each field opcode's declared position, so it cannot be cfg-gated"
    );
    let gated = fields.iter().filter(|field| field.cfg.is_some()).count();
    assert!(
        gated <= 1
            && fields
                .last()
                .is_some_and(|field| field.cfg.is_some() || gated == 0),
        "at most one field may be cfg-gated, and it must be the last"
    );
}

/// One field's copy of a field opcode template.
fn instantiate(template: &OpCodeDef, index: usize, field: &FieldInstance) -> OpCodeDef {
    let ty = &field.ty;
    OpCodeDef {
        name: format!("{}_{}", field.name, template.name),
        fn_name: template.fn_name.clone(),
        is_raw: template.is_raw,
        inputs: template.inputs.clone(),
        field_ty: quote! { #ty },
        generic: true,
        field: Some(index),
        cfg: field.cfg.clone(),
    }
}

/// The numbered opcode list: each template in declaration order, the first field's copy standing in
/// for a field opcode, then every further field's copies as one block per field.
fn expand_field_opcodes(templates: &[OpCodeDef], fields: &[FieldInstance]) -> Vec<OpCodeDef> {
    let mut codes: Vec<OpCodeDef> = templates
        .iter()
        .map(|template| match template.field {
            Some(_) => instantiate(template, 0, &fields[0]),
            None => template.clone(),
        })
        .collect();
    for (index, field) in fields.iter().enumerate().skip(1) {
        codes.extend(
            templates
                .iter()
                .filter(|template| template.field.is_some())
                .map(|template| instantiate(template, index, field)),
        );
    }
    codes
}

fn expect_ident(pat: &syn::Pat) -> Ident {
    match pat {
        Pat::Ident(ident) => ident.ident.clone(),
        _ => panic!("expected an identifier"),
    }
}

fn parse_signature(is_raw: bool, is_field: bool, sig: &syn::Signature) -> OpCodeDef {
    let name = sig.ident.to_string();
    let type_params = sig.generics.type_params().collect::<Vec<_>>();
    // The name a field operand's type is spelled with: the plain `Field`, or a field opcode's
    // type parameter.
    let field_name = if is_field {
        assert!(
            type_params.len() == 1,
            "field opcode `{name}` must be generic over exactly one type parameter, its field"
        );
        type_params[0].ident.to_string()
    } else {
        assert!(
            type_params.is_empty(),
            "opcode `{name}` is generic, so it must be a #[field_opcode]"
        );
        "Field".to_string()
    };

    let mut inputs = vec![];
    let skip = if is_raw { 3 } else { 0 };
    for arg in sig.inputs.iter().skip(skip) {
        match arg {
            FnArg::Receiver(_) => {
                panic!("receiver arguments are not allowed in the interpreter definition")
            }
            FnArg::Typed(pat) => {
                let input = parse_pat_type(pat, &field_name);
                inputs.push(input);
            }
        }
    }
    let def = OpCodeDef {
        fn_name: name.clone(),
        name,
        is_raw,
        inputs,
        field_ty: quote! { Field },
        generic: false,
        field: is_field.then_some(0),
        cfg: None,
    };
    check_slice_widths(&def);
    def
}

/// Every frame slice names a width parameter that is a host `u64` declared ahead of it.
///
/// Getters consume the instruction stream with a running offset in declaration order, so a width
/// declared after the slice that reads it has no value yet, and the generated code would not
/// compile.
fn check_slice_widths(def: &OpCodeDef) {
    let mut seen: Vec<&str> = Vec::new();
    for input in def.struct_args() {
        if let StructInputType::Slice(slice) = &input.val {
            assert!(
                seen.contains(&slice.width.as_str()),
                "opcode `{}`: frame slice `{}` takes its length from `{}`, which must be a host parameter declared before it",
                def.name,
                input.name,
                slice.width
            );
        }
        if matches!(input.val, StructInputType::Host(HostType::U64)) {
            seen.push(&input.name);
        }
    }
}

fn parse_pat_type(pat: &syn::PatType, field_name: &str) -> Input {
    if pat.attrs.len() > 1 {
        panic!("only one attribute is allowed in the interpreter definition");
    }

    if pat.attrs.is_empty() {
        return parse_unannotated(expect_ident(&pat.pat), pat.ty.as_ref(), field_name);
    }

    if pat.attrs[0].path().is_ident("out") {
        parse_out(expect_ident(&pat.pat), pat.ty.as_ref(), field_name)
    } else if pat.attrs[0].path().is_ident("frame") {
        parse_frame(expect_ident(&pat.pat), pat.ty.as_ref(), field_name)
    } else if pat.attrs[0].path().is_ident("frame_slice") {
        parse_frame_slice(expect_ident(&pat.pat), pat.ty.as_ref(), &pat.attrs[0])
    } else {
        panic!("unknown attribute");
    }
}

fn parse_unannotated(ident: Ident, ty: &syn::Type, field_name: &str) -> Input {
    match ty {
        syn::Type::Path(typath) => {
            let ty_ident = typath.path.require_ident().unwrap();
            if ty_ident == "Frame" {
                return Input::Frame;
            }
            Input::Struct(StructInput {
                name: ident.to_string(),
                val: StructInputType::Host(parse_host_type(ty, field_name)),
            })
        }
        syn::Type::Reference(intype) => match intype.elem.as_ref() {
            syn::Type::Path(typath) => {
                let ty_ident = typath.path.require_ident().unwrap();
                if ty_ident == "VM" {
                    Input::VM
                } else {
                    Input::Struct(StructInput {
                        name: ident.to_string(),
                        val: StructInputType::Host(parse_host_type(ty, field_name)),
                    })
                }
            }
            _ => Input::Struct(StructInput {
                name: ident.to_string(),
                val: StructInputType::Host(parse_host_type(ty, field_name)),
            }),
        },
        _ => Input::Struct(StructInput {
            name: ident.to_string(),
            val: StructInputType::Host(parse_host_type(ty, field_name)),
        }),
    }
}

fn parse_out(ident: Ident, ty: &syn::Type, field_name: &str) -> Input {
    match &ty {
        syn::Type::Ptr(ty) => {
            if ty.mutability.is_none() {
                panic!("out ref must be mutable");
            }
            Input::Struct(StructInput {
                name: ident.to_string(),
                val: StructInputType::Out(parse_guest_type(&ty.elem, field_name)),
            })
        }
        _ => {
            panic!("out must be a ptr");
        }
    }
}

fn parse_frame(ident: Ident, ty: &syn::Type, field_name: &str) -> Input {
    Input::Struct(StructInput {
        name: ident.to_string(),
        val: StructInputType::Frame(parse_guest_type(ty, field_name)),
    })
}

/// Parse `#[frame_slice(width)] name: &[u64]`, or `&mut [u64]` for a result. Both encode as one
/// `FramePosition`; the mutability says only what the handler may do with the cells.
fn parse_frame_slice(ident: Ident, ty: &syn::Type, attr: &syn::Attribute) -> Input {
    let width: Ident = attr
        .parse_args()
        .expect("frame_slice must name the width parameter, as #[frame_slice(bits)]");

    let syn::Type::Reference(reference) = ty else {
        panic!("a frame_slice must be a reference to a slice of u64, as &[u64] or &mut [u64]");
    };
    let syn::Type::Slice(slice) = reference.elem.as_ref() else {
        panic!("a frame_slice must be a slice, as &[u64] or &mut [u64]");
    };
    let syn::Type::Path(element) = slice.elem.as_ref() else {
        panic!("a frame_slice must hold u64, the frame's own cell");
    };
    assert!(
        element.path.is_ident("u64"),
        "a frame_slice must hold u64, the frame's own cell"
    );

    Input::Struct(StructInput {
        name: ident.to_string(),
        val: StructInputType::Slice(SliceInput {
            width: width.to_string(),
            mutable: reference.mutability.is_some(),
        }),
    })
}

fn parse_guest_type(ty: &syn::Type, field_name: &str) -> GuestType {
    match ty {
        syn::Type::Path(ty) => {
            let ident = ty.path.require_ident().unwrap();
            match ident.to_string().as_str() {
                name if name == field_name => GuestType::Field,
                "u64" => GuestType::U64,
                "Int128" => GuestType::Int128,
                "BoxedValue" => GuestType::BoxedValue,
                _ => panic!("unsupported guest path type {:?}", ty),
            }
        }
        syn::Type::Ptr(ty) => {
            if ty.mutability.is_none() {
                panic!("ptr must be mutable");
            }
            GuestType::Ptr
        }
        _ => {
            panic!("unsupported guest type {:?}", ty);
        }
    }
}

fn parse_host_type(ty: &syn::Type, field_name: &str) -> HostType {
    match ty {
        syn::Type::Path(ty) => {
            let ident = ty.path.require_ident().unwrap();
            match ident.to_string().as_str() {
                name if name == field_name => HostType::Field,
                "u64" => HostType::U64,
                "usize" => HostType::USize,
                "isize" => HostType::ISize,
                "JumpTarget" => HostType::JumpTarget,
                "FramePosition" => HostType::FramePosition,
                "BoxedLayout" => HostType::BoxedLayout,
                _ => panic!("unsupported type {:?}", ty),
            }
        }
        syn::Type::Reference(ty) => {
            assert!(
                ty.mutability.is_none(),
                "parameter reference must be immutable"
            );
            match ty.elem.as_ref() {
                syn::Type::Slice(ty) => {
                    HostType::Slice(Box::new(parse_host_type(&ty.elem, field_name)))
                }
                _ => panic!("References not supported here"),
            }
        }
        syn::Type::Tuple(ty) => HostType::Tuple(
            ty.elems
                .iter()
                .map(|ty| parse_host_type(ty, field_name))
                .collect(),
        ),
        _ => {
            panic!("unsupported host type {:?}", ty);
        }
    }
}

fn gen_handler(idx: usize, def: &OpCodeDef) -> proc_macro2::TokenStream {
    let prof_idx = idx;
    let field_ty = &def.field_ty;
    let mut getters = proc_macro2::TokenStream::new();
    for input in &def.inputs {
        match input {
            Input::Struct(StructInput {
                name,
                val: StructInputType::Frame(tp),
            }) => {
                let i = format_ident!("{}", name);
                let getter = match tp {
                    GuestType::Field => quote! { read_field::<#field_ty> },
                    GuestType::U64 => quote! { read_u64 },
                    GuestType::Int128 => quote! { read_int128 },
                    GuestType::Ptr => quote! { read_ptr },
                    GuestType::BoxedValue => quote! { read_array },
                };
                getters.extend(quote! {
                    let #i = unsafe {
                        frame.#getter(*pc.offset(current_field_offset) as isize)
                    };
                    current_field_offset += 1;
                });
            }
            Input::Struct(StructInput {
                name,
                val: StructInputType::Out(tp),
            }) => {
                let i: Ident = format_ident!("{}", name);
                let getter = match tp {
                    GuestType::Field => quote! { read_field_mut::<#field_ty> },
                    GuestType::U64 => quote! { read_u64_mut },
                    GuestType::Int128 => quote! { read_int128_mut },
                    GuestType::Ptr => quote! { read_ptr_mut },
                    GuestType::BoxedValue => quote! { read_array_mut },
                };
                // Bound, not read inline, so `gen_slice_bindings` can check a wide result against
                // it before any slice exists. A raw pointer may coexist with a slice; a write
                // through it may not.
                let pos = format_ident!("__out_pos_{}", name);
                getters.extend(quote! {
                    let #pos = unsafe { *pc.offset(current_field_offset) as isize };
                    let #i = unsafe { frame.#getter(#pos) };
                    current_field_offset += 1;
                });
            }
            Input::Struct(StructInput {
                name,
                val: StructInputType::Slice(slice),
            }) => {
                // A slice's position and length are bound here, in stream order, but the slice
                // itself is not built until every position is known — see `slice_bindings`.
                let pos = format_ident!("__slice_pos_{}", name);
                let len = format_ident!("__slice_len_{}", name);
                let width = format_ident!("{}", slice.width);
                getters.extend(quote! {
                    let #pos = unsafe { *pc.offset(current_field_offset) as isize };
                    current_field_offset += 1;
                    let #len = Frame::int_cells(#width);
                });
            }
            Input::Struct(StructInput {
                name,
                val: StructInputType::Host(htp),
            }) => {
                let i = format_ident!("{}", name);
                let offset_id = format_ident!("current_field_offset");
                let pc = format_ident!("pc");
                let getter = htp.make_getter(offset_id, pc, field_ty);
                getters.extend(quote! {
                    let #i = #getter;
                });
            }
            _ => {}
        }
    }

    let slice_bindings = gen_slice_bindings(def);

    let mut call_params = if def.is_raw {
        vec![quote! {pc}, quote! {frame}, quote! {vm}]
    } else {
        vec![]
    };

    call_params.extend(
        def.inputs
            .iter()
            .map(|input| match input {
                // Every operand kind is bound above to a local named after the operand, so the call
                // passes that name whatever the kind was.
                Input::Struct(StructInput { name, .. }) => {
                    let i = format_ident!("{}", name);
                    quote! { #i }
                }
                Input::VM => quote! { vm },
                Input::Frame => quote! { frame },
            })
            .collect::<Vec<_>>(),
    );
    let fn_name = format_ident!("{}", &def.fn_name);
    let op_name = if def.generic {
        quote! { #fn_name::<#field_ty> }
    } else {
        quote! { #fn_name }
    };

    let (call_and_finish, return_call) = if def.is_raw {
        // Raw opcodes already return `(*const u64, Frame)` themselves; the
        // wrapper just forwards that result.
        let call_op = quote! { #op_name(#(#call_params),*) };
        (quote! {}, call_op)
    } else {
        // Non-raw opcodes do their work and then advance the pc past their
        // own operands. The wrapper assembles the `(pc, frame)` pair for the
        // next dispatch step.
        let call_op = quote! { #op_name(#(#call_params),*); };
        let finish = quote! {
            let pc = unsafe { pc.offset(current_field_offset) };
            (pc, frame)
        };
        (call_op, finish)
    };

    let check_field = if def.generic {
        quote! { vm.check_field::<#field_ty>(); }
    } else {
        quote! {}
    };

    let handler_name = format_ident!("{}_handler", def.name);
    let prof_idx = prof_idx;
    let cfg = cfg_attr(&def.cfg);
    quote! {
        #cfg
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        pub fn #handler_name(
            pc: *const u64,
            frame: Frame,
            vm: &mut VM,
        ) -> (*const u64, Frame) {
            // unsafe {
            //     println!("opcode: {:?}", *(pc as *mut (*mut usize)));
            // }
            #[cfg(feature = "vm-profile")]
            let __prof_start = read_cycles();
            if vm.instruction_profile_enabled() {
                vm.record_instruction(pc, frame);
            }
            #check_field
            let mut current_field_offset = 1isize;
            #getters
            #slice_bindings
            #call_and_finish

            let __prof_ret = { #return_call };
            #[cfg(feature = "vm-profile")]
            {
                let __prof_elapsed = read_cycles().wrapping_sub(__prof_start);
                vm.opcode_profile[#prof_idx].0 += 1;
                vm.opcode_profile[#prof_idx].1 += __prof_elapsed;
            }
            __prof_ret
        }
    }
}

/// Build every frame slice a handler takes, after asserting that nothing the handler writes
/// overlaps one.
///
/// The check runs while there are only positions and lengths: a `&mut [u64]` and a `&[u64]` over
/// the same cells are undefined behavior as soon as both exist. A result slice is checked against
/// the other slices, and an `#[out]` against all of them, with its extent from
/// [`GuestType::to_cells_tokens`]. An `#[out]` is a raw pointer, which may coexist with a slice,
/// but the handler body writes through it while every binding is live.
fn gen_slice_bindings(def: &OpCodeDef) -> proc_macro2::TokenStream {
    let slices = def
        .struct_args()
        .filter_map(|input| match &input.val {
            StructInputType::Slice(slice) => Some((input.name.as_str(), slice)),
            _ => None,
        })
        .collect::<Vec<_>>();

    // Each `#[out]` as the pair of expressions the overlap test needs: where it starts and how far
    // it reaches.
    let outs = def
        .struct_args()
        .filter_map(|input| match &input.val {
            StructInputType::Out(tp) => {
                let pos = format_ident!("__out_pos_{}", input.name);
                Some((
                    input.name.as_str(),
                    quote! { #pos },
                    tp.to_cells_tokens(&def.field_ty),
                ))
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    let extents = slices
        .iter()
        .map(|(name, slice)| {
            let (pos, len) = (
                format_ident!("__slice_pos_{}", name),
                format_ident!("__slice_len_{}", name),
            );
            (*name, slice.mutable, quote! { #pos }, quote! { #len })
        })
        .collect::<Vec<_>>();

    // Each writer is checked against every slice. Two `#[out]`s are not checked against each
    // other, as neither is a slice.
    let writers = extents
        .iter()
        .filter(|(_, mutable, _, _)| *mutable)
        .map(|(name, _, pos, len)| (*name, pos.clone(), len.clone()))
        .chain(outs)
        .collect::<Vec<_>>();

    let mut result = proc_macro2::TokenStream::new();
    for (name, pos, len) in &writers {
        for (other, _, other_pos, other_len) in &extents {
            if other == name {
                continue;
            }
            let message = format!(
                "frame operands `{name}` and `{other}` overlap, so writing one would corrupt the \
                 other"
            );
            result.extend(quote! {
                debug_assert!(
                    #pos + (#len as isize) <= #other_pos
                        || #other_pos + (#other_len as isize) <= #pos,
                    #message
                );
            });
        }
    }

    for (name, slice) in &slices {
        let i = format_ident!("{}", name);
        let (pos, len) = (
            format_ident!("__slice_pos_{}", name),
            format_ident!("__slice_len_{}", name),
        );
        let reader = if slice.mutable {
            format_ident!("read_limbs_mut")
        } else {
            format_ident!("read_limbs")
        };
        result.extend(quote! {
            let #i = unsafe { frame.#reader(#pos, #len) };
        });
    }

    result
}

/// The `name: Type` of each operand of `code`, as an `OpCode` variant holds it.
fn variant_params(code: &OpCodeDef) -> Vec<proc_macro2::TokenStream> {
    code.struct_args()
        .map(|input| {
            let i = format_ident!("{}", input.name);
            match &input.val {
                StructInputType::Frame(_) => quote! {#i: FramePosition},
                StructInputType::Out(_) => quote! {#i: FramePosition},
                StructInputType::Slice(_) => quote! {#i: FramePosition},
                StructInputType::Host(ty) => {
                    let ty = ty.to_def_tokens(&code.field_ty);
                    quote! {#i: #ty}
                }
            }
        })
        .collect()
}

fn gen_opcode_enum(codes: &[OpCodeDef]) -> proc_macro2::TokenStream {
    let codes = codes.iter().map(|code| {
        let params = variant_params(code);
        let name = format_ident!("{}", code.struct_name());
        let cfg = cfg_attr(&code.cfg);
        quote! {
            #cfg
            #name{#(#params),*}
        }
    });

    quote! {
        pub enum OpCode {
            #(#codes),*
        }
    }
}

fn gen_opcode_helpers(codes: &[OpCodeDef]) -> proc_macro2::TokenStream {
    let opcode_writer_cases = codes
        .iter()
        .enumerate()
        .map(|(i, code)| {
            let i = i as u64;
            let name = format_ident!("{}", code.struct_name());
            let params = code
                .struct_args()
                .map(|input| format_ident!("{}", input.name))
                .collect::<Vec<_>>();
            let param_writes = code
                .struct_args()
                .map(|param| {
                    let i = format_ident!("{}", param.name);
                    match &param.val {
                        StructInputType::Frame(_) => {
                            let r = quote! {
                                binary.push(#i.0 as u64);
                                init_offset -= 1;
                            };
                            r
                        }
                        StructInputType::Out(_) | StructInputType::Slice(_) => {
                            let r = quote! {
                                binary.push(#i.0 as u64);
                                init_offset -= 1;
                            };
                            r
                        }
                        StructInputType::Host(htp) => {
                            let offset = format_ident!("init_offset");
                            let binary = format_ident!("binary");
                            let jumps_to_fix = format_ident!("jumps_to_fix");
                            htp.make_serializer(
                                i.to_token_stream(),
                                &offset,
                                &binary,
                                &jumps_to_fix,
                                true,
                                &code.field_ty,
                            )
                        }
                    }
                })
                .collect::<Vec<_>>();
            let cfg = cfg_attr(&code.cfg);
            quote! {
                #cfg
                Self::#name { #(#params),* } => {
                    binary.push(#i);
                    let mut init_offset = -1isize;
                    #(#param_writes)*
                }
            }
        })
        .collect::<Vec<_>>();

    let opcode_measure_cases = codes.iter().enumerate().map(|(i, code)| {
        let i = i as u64;
        let mut result = proc_macro2::TokenStream::new();
        for inp in code.struct_args() {
            match &inp.val {
                StructInputType::Frame(_) => {
                    result.extend(quote! {
                        ix += 1;
                    });
                }
                StructInputType::Out(_) | StructInputType::Slice(_) => {
                    result.extend(quote! {
                        ix += 1;
                    });
                }
                StructInputType::Host(htp) => {
                    let binary = format_ident!("binary");
                    let ix = format_ident!("ix");
                    result.extend(htp.make_measurer(&binary, &ix, &code.field_ty));
                }
            }
        }
        let cfg = cfg_attr(&code.cfg);
        quote! {
            #cfg
            #i => {#result}
        }
    });

    let opcode_display_cases = codes.iter().map(|code| {
        let matcher = code.matcher();
        let name_str = code.name.clone();
        let fields_ident = format_ident!("fields");
        let fields = code
            .struct_args()
            .map(|field| field.make_printer(&fields_ident));
        let cfg = cfg_attr(&code.cfg);
        quote! {
            #cfg
            #matcher => {
                let mut #fields_ident = String::new();
                #(#fields);*
                write!(f, "{}{}", #name_str, #fields_ident).unwrap();
            }
        }
    });

    // The tables indexed by opcode number. A `cfg`-gated field's copies are the last opcodes (see
    // `check_fields`), so the tables without them are a prefix of the tables with them, and both
    // are generated, each under its side of the `cfg`.
    let tables = match codes.iter().find_map(|code| code.cfg.clone()) {
        None => gen_opcode_tables(codes, quote! {}),
        Some(cfg) => {
            let ungated = codes
                .iter()
                .take_while(|code| code.cfg.is_none())
                .cloned()
                .collect::<Vec<_>>();
            let mut tables = gen_opcode_tables(codes, quote! { #[cfg(#cfg)] });
            tables.extend(gen_opcode_tables(&ungated, quote! { #[cfg(not(#cfg))] }));
            tables
        }
    };

    quote! {
        #tables

        impl OpCode {
            pub fn to_binary(&self, binary: &mut Vec<u64>, jumps_to_fix: &mut Vec<(usize, isize)>) {
                match self {
                    #(#opcode_writer_cases),*
                }
            }

            pub fn next_opcode(binary: &[u64], mut ix: usize) -> usize {
                let op_code = binary[ix];
                ix += 1;
                match op_code {
                    #(#opcode_measure_cases),*
                    _ => panic!("unknown opcode"),
                }
                ix
            }
        }

        impl std::fmt::Display for OpCode {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    #(#opcode_display_cases),*
                }
                Ok(())
            }
        }
    }
}

/// The opcode-number-indexed tables over `codes`, each item carrying `cfg`.
fn gen_opcode_tables(
    codes: &[OpCodeDef],
    cfg: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let dsp_size = codes.len();
    let dispatch_cases = codes.iter().map(|code| {
        let name = format_ident!("{}_handler", code.name);
        quote! {
            #name
        }
    });

    let opcode_name_strs = codes.iter().map(|code| code.name.clone());
    let raw_opcode_indices = codes
        .iter()
        .enumerate()
        .filter_map(|(index, code)| code.is_raw.then_some(index));

    quote! {
        /// The number of distinct opcodes — i.e. the size of the dispatch table
        /// and of the VM's per-opcode profiling buffer.
        #cfg
        pub const NUM_OPCODES: usize = #dsp_size;

        /// Opcode names indexed by discriminant. Used by the profiling report and the pinned
        /// opcode-number test.
        #cfg
        #[cfg(any(test, feature = "vm-profile"))]
        pub static OPCODE_NAMES: [&str; #dsp_size] = [ #(#opcode_name_strs),* ];

        /// Lets dispatch tests require coverage when a new raw opcode is introduced.
        #cfg
        #[cfg(test)]
        pub const RAW_OPCODES: &[usize] = &[#(#raw_opcode_indices),*];

        #cfg
        pub const DISPATCH: [Handler; #dsp_size] = [ #(#dispatch_cases),* ];
    }
}

/// `FieldOpCode`, the field-independent form of every field opcode, and its `for_field`.
///
/// Codegen builds a `FieldOpCode` with named operands and picks the field once, through an
/// exhaustive `match` on `FieldId`, so a new `FieldId` variant is a compile error here. A fallback
/// arm would silently emit another field's opcode for it. `cfg`-gated fields have no `FieldId`
/// variant and so no arm.
fn gen_field_opcode_enum(
    templates: &[OpCodeDef],
    codes: &[OpCodeDef],
    fields: &[FieldInstance],
) -> proc_macro2::TokenStream {
    let families = templates
        .iter()
        .filter(|template| template.field.is_some())
        .collect::<Vec<_>>();

    let variants = families.iter().map(|template| {
        assert!(
            !template.struct_args().any(
                |input| matches!(&input.val, StructInputType::Host(ty) if ty.contains_field())
            ),
            "field opcode `{}` takes a field immediate, which has no field-independent type for \
             `FieldOpCode` to hold",
            template.name
        );
        let name = format_ident!("{}", template.struct_name());
        let params = variant_params(template);
        quote! { #name { #(#params),* } }
    });

    // One method per field, named after it, choosing that field's copy.
    let choosers = fields.iter().enumerate().map(|(index, field)| {
        let arms = families.iter().map(|template| {
            let name = format_ident!("{}", template.struct_name());
            let args = template
                .struct_args()
                .map(|input| format_ident!("{}", input.name))
                .collect::<Vec<_>>();
            let code = codes
                .iter()
                .find(|code| code.fn_name == template.fn_name && code.field == Some(index))
                .expect("every field has a copy of every field opcode");
            let copy = format_ident!("{}", code.struct_name());
            quote! { FieldOpCode::#name { #(#args),* } => OpCode::#copy { #(#args),* } }
        });
        let method = &field.name;
        let cfg = cfg_attr(&field.cfg);
        let doc = format!("The opcode performing this operation over `{method}`.");
        quote! {
            #cfg
            #[doc = #doc]
            pub fn #method(self) -> OpCode {
                match self {
                    #(#arms),*
                }
            }
        }
    });

    // A field behind a `cfg` has no `FieldId`.
    let field_arms = fields
        .iter()
        .filter(|field| field.cfg.is_none())
        .map(|field| {
            let id = format_ident!("{}", field.name.to_string().to_pascal_case());
            let method = &field.name;
            quote! { ::mavros_artifacts::FieldId::#id => self.#method() }
        });

    quote! {
        /// A field opcode with its field not yet chosen; [`FieldOpCode::for_field`] chooses it.
        pub enum FieldOpCode {
            #(#variants),*
        }

        impl FieldOpCode {
            #(#choosers)*

            /// The opcode performing this operation over `field`.
            pub fn for_field(self, field: ::mavros_artifacts::FieldId) -> OpCode {
                match field {
                    #(#field_arms),*
                }
            }
        }
    }
}
