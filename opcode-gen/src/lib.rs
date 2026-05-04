use proc_macro::TokenStream;
use quote::ToTokens;
use quote::format_ident;
use quote::quote;
use stringcase::Caser;
use syn::{FnArg, Ident, Pat, parse_macro_input};

#[derive(Debug)]
enum GuestType {
    Field,
    U64,
    Ptr,
    BoxedValue,
}

#[derive(Debug)]
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
    fn to_def_tokens(&self) -> proc_macro2::TokenStream {
        match self {
            HostType::U64 => quote! { u64 },
            HostType::USize => quote! { usize },
            HostType::ISize => quote! { isize },
            HostType::JumpTarget => quote! { JumpTarget },
            HostType::Field => quote! { Field },
            HostType::Slice(ty) => {
                let child = ty.to_def_tokens();
                quote! { Vec<#child> }
            }
            HostType::Tuple(ty) => {
                let children = ty.iter().map(|ty| ty.to_def_tokens()).collect::<Vec<_>>();
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
                let size = ty.measure_size() as isize;
                quote! {
                    unsafe {
                        let len = *#pc.offset(#offset) as usize;
                        #offset += 1;
                        let r = std::slice::from_raw_parts(
                            #pc.offset(#offset) as *const _,
                            len
                        );
                        #offset += len as isize * #size;
                        r
                    }
                }
            }
            HostType::Field => {
                quote! {
                    unsafe {
                        let r0 = *#pc.offset(#offset);
                        let r1 = *#pc.offset(#offset + 1);
                        let r2 = *#pc.offset(#offset + 2);
                        let r3 = *#pc.offset(#offset + 3);
                        #offset += 4;
                        ark_ff::Fp(ark_ff::BigInt([r0, r1, r2, r3]), std::marker::PhantomData)
                    }
                }
            }
            HostType::Tuple(_ty) => {
                todo!("tuple getter");
            }
        }
    }

    fn measure_size(&self) -> usize {
        match self {
            HostType::U64
            | HostType::USize
            | HostType::ISize
            | HostType::JumpTarget
            | HostType::FramePosition
            | HostType::BoxedLayout => 1,
            HostType::Field => 4, // TODO parameterize
            HostType::Slice(_) => panic!("slice size is not known at compile time"),
            HostType::Tuple(ty) => ty.iter().map(|ty| ty.measure_size()).sum(),
        }
    }

    fn make_serializer(
        &self,
        i: proc_macro2::TokenStream,
        offset: &proc_macro2::Ident,
        binary: &proc_macro2::Ident,
        jumps_to_fix: &proc_macro2::Ident,
        is_ref: bool,
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
                quote! {
                    #binary.push(#i.0.0[0] as u64);
                    #binary.push(#i.0.0[1] as u64);
                    #binary.push(#i.0.0[2] as u64);
                    #binary.push(#i.0.0[3] as u64);
                    #offset -= 1;
                }
            }
            HostType::Slice(intp) => {
                let ixed = quote! { elem };
                let child_serializer =
                    intp.make_serializer(ixed.clone(), offset, binary, jumps_to_fix, true);
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
                    result.extend(intp.make_serializer(ixed, offset, binary, jumps_to_fix, false));
                }
                result
            }
        }
    }

    fn make_measurer(
        &self,
        binary: &proc_macro2::Ident,
        ix: &proc_macro2::Ident,
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
                quote! {
                    #ix += 4;
                }
            }
            HostType::Slice(child) => {
                let child_size = child.measure_size();
                quote! {
                    {
                        let len = #binary[#ix] as usize;
                        #ix += 1;
                        #ix += len * #child_size;
                    };
                }
            }
            HostType::Tuple(_children) => {
                quote! { todo!("tuple measurer") }
            }
        }
    }
}

#[derive(Debug)]
enum StructInputType {
    Frame(GuestType),
    Out(GuestType),
    Host(HostType),
}

#[derive(Debug)]
struct StructInput {
    name: String,
    val:  StructInputType,
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
            StructInputType::Out(_) => quote! {
                #fields_ident = format!("{} %{}", #fields_ident, #ident.0);
            },
            _ => quote! {
                #fields_ident = format!("{} _", #fields_ident);
            },
        }
    }
}

#[derive(Debug)]
enum Input {
    Struct(StructInput),
    Frame,
    VM,
}

#[derive(Debug)]
struct OpCodeDef {
    name:   String,
    is_raw: bool,
    inputs: Vec<Input>,
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

#[proc_macro_attribute]
pub fn interpreter(_attr: TokenStream, item: TokenStream) -> TokenStream {
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
                func.attrs = vec![];
                for attr in old_attrs {
                    if attr.path().is_ident("opcode") {
                        continue;
                    }
                    if attr.path().is_ident("raw_opcode") {
                        is_raw = true;
                        continue;
                    }
                    if attr.path().is_ident("inline") {
                        controls_inline = true;
                    }
                    func.attrs.push(attr);
                }

                let def = parse_signature(is_raw, &func.sig);
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

    for code in &codes {
        result.extend(gen_handler(code));
    }

    result.extend(gen_opcode_enum(&codes));
    result.extend(gen_opcode_helpers(&codes));

    // println!("codes: {:?}", codes);

    result.into()
}

fn expect_ident(pat: &syn::Pat) -> Ident {
    match pat {
        Pat::Ident(ident) => ident.ident.clone(),
        _ => panic!("expected an identifier"),
    }
}

fn parse_signature(is_raw: bool, sig: &syn::Signature) -> OpCodeDef {
    let mut inputs = vec![];
    let skip = if is_raw { 3 } else { 0 };
    for arg in sig.inputs.iter().skip(skip) {
        match arg {
            FnArg::Receiver(_) => {
                panic!("receiver arguments are not allowed in the interpreter definition")
            }
            FnArg::Typed(pat) => {
                let input = parse_pat_type(pat);
                inputs.push(input);
            }
        }
    }
    OpCodeDef {
        name: sig.ident.to_string(),
        is_raw,
        inputs,
    }
}

fn parse_pat_type(pat: &syn::PatType) -> Input {
    if pat.attrs.len() > 1 {
        panic!("only one attribute is allowed in the interpreter definition");
    }

    if pat.attrs.is_empty() {
        return parse_unannotated(expect_ident(&pat.pat), pat.ty.as_ref());
    }

    if pat.attrs[0].path().is_ident("out") {
        return parse_out(expect_ident(&pat.pat), pat.ty.as_ref());
    } else if pat.attrs[0].path().is_ident("frame") {
        return parse_frame(expect_ident(&pat.pat), pat.ty.as_ref());
    } else {
        panic!("unknown attribute");
    }
}

fn parse_unannotated(ident: Ident, ty: &syn::Type) -> Input {
    match ty {
        syn::Type::Path(typath) => {
            let ty_ident = typath.path.require_ident().unwrap();
            if ty_ident == "Frame" {
                return Input::Frame;
            }
            return Input::Struct(StructInput {
                name: ident.to_string(),
                val:  StructInputType::Host(parse_host_type(ty)),
            });
        }
        syn::Type::Reference(intype) => match intype.elem.as_ref() {
            syn::Type::Path(typath) => {
                let ty_ident = typath.path.require_ident().unwrap();
                if ty_ident == "VM" {
                    return Input::VM;
                } else {
                    return Input::Struct(StructInput {
                        name: ident.to_string(),
                        val:  StructInputType::Host(parse_host_type(ty)),
                    });
                }
            }
            _ => {
                return Input::Struct(StructInput {
                    name: ident.to_string(),
                    val:  StructInputType::Host(parse_host_type(ty)),
                });
            }
        },
        _ => {
            return Input::Struct(StructInput {
                name: ident.to_string(),
                val:  StructInputType::Host(parse_host_type(ty)),
            });
        }
    }
}

fn parse_out(ident: Ident, ty: &syn::Type) -> Input {
    match &ty {
        syn::Type::Ptr(ty) => {
            if ty.mutability.is_none() {
                panic!("out ref must be mutable");
            }
            return Input::Struct(StructInput {
                name: ident.to_string(),
                val:  StructInputType::Out(parse_guest_type(&ty.elem)),
            });
        }
        _ => {
            panic!("out must be a ptr");
        }
    }
}

fn parse_frame(ident: Ident, ty: &syn::Type) -> Input {
    return Input::Struct(StructInput {
        name: ident.to_string(),
        val:  StructInputType::Frame(parse_guest_type(ty)),
    });
}

fn parse_guest_type(ty: &syn::Type) -> GuestType {
    match ty {
        syn::Type::Path(ty) => {
            let ident = ty.path.require_ident().unwrap();
            return match ident.to_string().as_str() {
                "Field" => GuestType::Field,
                "u64" => GuestType::U64,
                "BoxedValue" => GuestType::BoxedValue,
                _ => panic!("unsupported guest path type {:?}", ty),
            };
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

fn parse_host_type(ty: &syn::Type) -> HostType {
    match ty {
        syn::Type::Path(ty) => {
            let ident = ty.path.require_ident().unwrap();
            return match ident.to_string().as_str() {
                "u64" => HostType::U64,
                "usize" => HostType::USize,
                "isize" => HostType::ISize,
                "JumpTarget" => HostType::JumpTarget,
                "FramePosition" => HostType::FramePosition,
                "BoxedLayout" => HostType::BoxedLayout,
                "Field" => HostType::Field,
                _ => panic!("unsupported type {:?}", ty),
            };
        }
        syn::Type::Reference(ty) => {
            assert!(
                ty.mutability.is_none(),
                "parameter reference must be immutable"
            );
            match ty.elem.as_ref() {
                syn::Type::Slice(ty) => HostType::Slice(Box::new(parse_host_type(&ty.elem))),
                _ => panic!("References not supported here"),
            }
        }
        syn::Type::Tuple(ty) => {
            HostType::Tuple(ty.elems.iter().map(|elem| parse_host_type(elem)).collect())
        }
        _ => {
            panic!("unsupported host type {:?}", ty);
        }
    }
}

fn gen_handler(def: &OpCodeDef) -> proc_macro2::TokenStream {
    let mut getters = proc_macro2::TokenStream::new();
    for input in &def.inputs {
        match input {
            Input::Struct(StructInput {
                name,
                val: StructInputType::Frame(tp),
            }) => {
                let i = format_ident!("{}", name);
                let getter = match tp {
                    GuestType::Field => format_ident!("read_field"),
                    GuestType::U64 => format_ident!("read_u64"),
                    GuestType::Ptr => format_ident!("read_ptr"),
                    GuestType::BoxedValue => format_ident!("read_array"),
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
                    GuestType::Field => format_ident!("read_field_mut"),
                    GuestType::U64 => format_ident!("read_u64_mut"),
                    GuestType::Ptr => format_ident!("read_ptr_mut"),
                    GuestType::BoxedValue => format_ident!("read_array_mut"),
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
                val: StructInputType::Host(htp),
            }) => {
                let i = format_ident!("{}", name);
                let offset_id = format_ident!("current_field_offset");
                let pc = format_ident!("pc");
                let getter = htp.make_getter(offset_id, pc);
                getters.extend(quote! {
                    let #i = #getter;
                });
            }
            _ => {}
        }
    }

    let mut call_params = if def.is_raw {
        vec![quote! {pc}, quote! {frame}, quote! {vm}]
    } else {
        vec![]
    };

    call_params.extend(
        def.inputs
            .iter()
            .map(|input| match input {
                Input::Struct(StructInput {
                    name,
                    val: StructInputType::Frame(_),
                }) => {
                    let i = format_ident!("{}", name);
                    quote! { #i }
                }
                Input::Struct(StructInput {
                    name,
                    val: StructInputType::Out(_),
                }) => {
                    let i = format_ident!("{}", name);
                    quote! { #i }
                }
                Input::Struct(StructInput {
                    name,
                    val: StructInputType::Host(_),
                }) => {
                    let i = format_ident!("{}", name);
                    quote! {#i}
                }
                Input::VM => quote! { vm },
                Input::Frame => quote! { frame },
            })
            .collect::<Vec<_>>(),
    );
    let op_name = format_ident!("{}", &def.name);

    let call_op = quote! {
        #op_name(#(#call_params),*);
    };

    let finish = if def.is_raw {
        quote! {}
    } else {
        quote! {
            let pc = unsafe { pc.offset(current_field_offset) };
            dispatch(pc, frame, vm)
        }
    };

    let handler_name = format_ident!("{}_handler", def.name);
    quote! {
        pub fn #handler_name(
            pc: *const u64,
            frame: Frame,
            vm: &mut VM,
        ) {
            // unsafe {
            //     println!("opcode: {:?}", *(pc as *mut (*mut usize)));
            // }
            let mut current_field_offset = 1isize;
            #getters
            #call_op

            #finish
        }
    }
}

fn gen_opcode_enum(codes: &[OpCodeDef]) -> proc_macro2::TokenStream {
    let codes = codes.iter().map(|code| {
        let params = code
            .struct_args()
            .map(|input| {
                let i = format_ident!("{}", input.name);
                match &input.val {
                    // StructInputType::Const(_) => todo!(),
                    StructInputType::Frame(_) => Some(quote! {#i: FramePosition}),
                    StructInputType::Out(_) => Some(quote! {#i: FramePosition}),
                    StructInputType::Host(ty) => {
                        let ty = ty.to_def_tokens();
                        Some(quote! {#i: #ty})
                    }
                }
            })
            .collect::<Vec<_>>();
        let name = format_ident!("{}", &code.name.to_string().to_pascal_case());
        quote! {
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
                        StructInputType::Out(_) => {
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
                            )
                        }
                    }
                })
                .collect::<Vec<_>>();
            quote! {
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
                StructInputType::Out(_) => {
                    result.extend(quote! {
                        ix += 1;
                    });
                }
                StructInputType::Host(htp) => {
                    let binary = format_ident!("binary");
                    let ix = format_ident!("ix");
                    result.extend(htp.make_measurer(&binary, &ix));
                }
            }
        }
        quote! {
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
        quote! {
            #matcher => {
                let mut #fields_ident = String::new();
                #(#fields);*
                write!(f, "{}{}", #name_str, #fields_ident).unwrap();
            }
        }
    });

    let dsp_size = codes.len();
    let dispatch_cases = codes.iter().map(|code| {
        let name = format_ident!("{}_handler", code.name);
        quote! {
            #name
        }
    });

    quote! {
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

        pub const DISPATCH: [Handler; #dsp_size] = [ #(#dispatch_cases),* ];
    }
}
