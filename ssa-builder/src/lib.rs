use proc_macro::TokenStream;
use quote::ToTokens;
use quote::format_ident;
use quote::quote;
use syn::Expr;
use syn::braced;
use syn::parenthesized;
use syn::parse::Parse;
use syn::parse::ParseStream;
use syn::punctuated::Punctuated;
use syn::{Ident, parse_macro_input};

#[derive(Debug)]
enum Operand {
    Captured(Ident),
    Fresh(Ident),
    UConst(usize, Expr),
    FieldConst(Expr),
}

#[derive(Debug)]
struct Instruction {
    name: Ident,
    operands: Vec<Operand>,
}

struct ResultOperand(Operand);

struct InputOperand(Operand);

#[derive(Debug)]
struct Snippet {
    func_expr: syn::Expr,
    instructions: Vec<Instruction>,
    results: Vec<syn::Ident>,
}

mod puncts {
    syn::custom_punctuation!(Let, :=);
    syn::custom_punctuation!(Const, !);
    syn::custom_punctuation!(CapturedResult, #);
}
use puncts::{CapturedResult, Const, Let};

impl Parse for InputOperand {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(Const) {
            input.parse::<Const>()?;
            let expr = input.parse::<Expr>()?;
            input.parse::<syn::Token![:]>()?;
            let tp = input.parse::<syn::Ident>()?;
            match tp.to_string().as_str() {
                "u32" => Ok(InputOperand(Operand::UConst(32, expr))),
                "Field" => Ok(InputOperand(Operand::FieldConst(expr))),
                _ => panic!("unknown type"),
            }
        } else {
            input.parse().map(|r| InputOperand(Operand::Captured(r)))
        }
    }
}

impl Parse for ResultOperand {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(CapturedResult) {
            input.parse::<CapturedResult>()?;
            input.parse().map(|i| ResultOperand(Operand::Captured(i)))
        } else {
            input.parse().map(|r| ResultOperand(Operand::Fresh(r)))
        }
    }
}

fn is_assignment(input: ParseStream) -> syn::Result<()> {
    let fork: syn::parse::ParseBuffer<'_> = input.fork();
    Punctuated::<ResultOperand, syn::Token![,]>::parse_separated_nonempty(&fork)?;
    fork.parse::<Let>()?;
    Ok(())
}

impl Parse for Instruction {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let results = if is_assignment(input).is_ok() {
            let inputs =
                Punctuated::<ResultOperand, syn::Token![,]>::parse_separated_nonempty(&input)?;
            input.parse::<Let>()?;
            inputs
        } else {
            Punctuated::new()
        };
        let instruction_name = input.parse::<Ident>()?;
        let args;
        parenthesized!(args in input);
        let args = args.parse_terminated(InputOperand::parse, syn::Token![,])?;
        let operands = results
            .into_iter()
            .map(|r| r.0)
            .chain(args.into_iter().map(|a| a.0))
            .collect::<Vec<_>>();
        Ok(Instruction {
            name: instruction_name,
            operands: operands,
        })
    }
}

impl Parse for Snippet {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let func = input.parse::<syn::Expr>()?;
        input.parse::<syn::Token![,]>()?;
        let instructions;
        braced!(instructions in input);

        input.parse::<syn::Token![->]>()?;

        let results = input.parse_terminated(Ident::parse, syn::Token![,])?;

        let instructions = instructions.parse_terminated(Instruction::parse, syn::Token![;])?;

        Ok(Snippet {
            func_expr: func,
            instructions: instructions.into_iter().collect::<Vec<_>>(),
            results: results.into_iter().collect::<Vec<_>>(),
        })
    }
}

struct AppendSnippet {
    func_expr: syn::Expr,
    result: syn::Expr,
    instructions: Vec<Instruction>,
    results: Vec<syn::Ident>,
}

impl Parse for AppendSnippet {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let func = input.parse::<syn::Expr>()?;
        input.parse::<syn::Token![,]>()?;
        let result = input.parse::<syn::Expr>()?;
        input.parse::<syn::Token![,]>()?;
        let instructions;
        braced!(instructions in input);

        input.parse::<syn::Token![->]>()?;

        let results = input.parse_terminated(Ident::parse, syn::Token![,])?;

        let instructions = instructions.parse_terminated(Instruction::parse, syn::Token![;])?;

        Ok(AppendSnippet {
            func_expr: func,
            result: result,
            instructions: instructions.into_iter().collect::<Vec<_>>(),
            results: results.into_iter().collect::<Vec<_>>(),
        })
    }
}

#[proc_macro]
pub fn ssa_snippet(input: TokenStream) -> TokenStream {
    let snippet = parse_macro_input!(input as Snippet);

    let result_fields = snippet.results.iter().map(|r| quote! { #r: ValueId });

    let mut instructions = quote! {
        use crate::compiler::ssa::OpCode;
        use crate::compiler::taint_analysis::ConstantTaint;
        let mut __ssa_builder_result: Vec<OpCode<ConstantTaint>> = vec![];
        struct Result {
            #(#result_fields),*
        };
    };

    let func_expr = snippet.func_expr.to_token_stream();

    for instruction in snippet.instructions {
        let mut operands = vec![];
        for operand in instruction.operands {
            match operand {
                Operand::Fresh(ident) => {
                    instructions.extend(quote! {
                        let #ident = #func_expr.fresh_value();
                    });
                    operands.push(quote! { #ident });
                }
                Operand::Captured(ident) => {
                    operands.push(quote! { #ident });
                }
                Operand::UConst(size, expr) => {
                    operands.push(quote! { #func_expr.push_u_const(#size, #expr) });
                }
                Operand::FieldConst(expr) => {
                    operands.push(quote! { #func_expr.push_field_const(#expr) });
                }
            }
        }
        let instruction_name = format_ident!("mk_{}", instruction.name);
        instructions.extend(quote! {
            __ssa_builder_result.push(OpCode::#instruction_name(#(#operands),*));
        })
    }

    let results = snippet.results.iter();
    TokenStream::from(quote! {
        {
            #instructions
            (__ssa_builder_result, Result { #(#results),* })
        }
    })
}

#[proc_macro]
pub fn ssa_append(input: TokenStream) -> TokenStream {
    let snippet = parse_macro_input!(input as AppendSnippet);

    let result_fields = snippet.results.iter().map(|r| quote! { #r: ValueId });
    let accumulator = snippet.result.to_token_stream();

    let mut instructions = quote! {
        use crate::compiler::ssa::OpCode;
        use crate::compiler::taint_analysis::ConstantTaint;
        struct Result {
            #(#result_fields),*
        };
    };

    let func_expr = snippet.func_expr.to_token_stream();

    for instruction in snippet.instructions {
        let mut operands = vec![];
        for operand in instruction.operands {
            match operand {
                Operand::Fresh(ident) => {
                    instructions.extend(quote! {
                        let #ident = #func_expr.fresh_value();
                    });
                    operands.push(quote! { #ident });
                }
                Operand::Captured(ident) => {
                    operands.push(quote! { #ident });
                }
                Operand::UConst(size, expr) => {
                    operands.push(quote! { #func_expr.push_u_const(#size, #expr) });
                }
                Operand::FieldConst(expr) => {
                    operands.push(quote! { #func_expr.push_field_const(#expr) });
                }
            }
        }
        let instruction_name = format_ident!("mk_{}", instruction.name);
        instructions.extend(quote! {
            #accumulator.push(OpCode::#instruction_name(#(#operands),*));
        })
    }

    let results = snippet.results.iter();
    TokenStream::from(quote! {
        {
            #instructions
            Result { #(#results),* }
        }
    })
}
