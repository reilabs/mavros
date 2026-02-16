//! Converts monomorphized AST types to mavros SSA types.

use noirc_frontend::monomorphization::ast::Type as AstType;
use noirc_frontend::shared::Signedness;

use crate::compiler::ir::r#type::Type;

/// Converts AST types to SSA types.
pub struct TypeConverter;

impl TypeConverter {
    pub fn new() -> Self {
        Self
    }

    /// Convert a monomorphized AST type to an SSA type.
    pub fn convert_type(&self, ast_type: &AstType) -> Type {
        match ast_type {
            AstType::Field => Type::field(),
            AstType::Bool => Type::bool(),
            AstType::Integer(signedness, bit_size) => match signedness {
                Signedness::Unsigned => Type::u(bit_size.bit_size() as usize),
                Signedness::Signed => panic!("Signed integers not supported"),
            },
            AstType::Unit => {
                // Unit type is represented as an empty tuple
                Type::tuple_of(vec![])
            }
            AstType::Array(len, elem_type) => {
                let elem = self.convert_type(elem_type);
                elem.array_of(*len as usize)
            }
            AstType::Vector(elem_type) => {
                let elem = self.convert_type(elem_type);
                elem.slice_of()
            }
            AstType::Tuple(types) => {
                let converted: Vec<Type> = types.iter().map(|t| self.convert_type(t)).collect();
                Type::tuple_of(converted)
            }
            AstType::Reference(inner, _mutable) => {
                let inner_type = self.convert_type(inner);
                inner_type.ref_of()
            }
            AstType::Function(_, _, _, _) => Type::function(),
            AstType::String(len) => {
                // str<N>: N is UTF-8 byte count, represented as Array(U(8), N)
                Type::u(8).array_of(*len as usize)
            }
            AstType::FmtString(len, captures_type) => {
                // fmtstr<N, T>: N is codepoint count, represented as
                // Tuple(Array(U(32), N), ...T_fields)
                let codepoints_array = Type::u(32).array_of(*len as usize);
                let capture_fields = match captures_type.as_ref() {
                    AstType::Tuple(fields) => fields
                        .iter()
                        .map(|t| self.convert_type(t))
                        .collect::<Vec<_>>(),
                    AstType::Unit => vec![],
                    other => vec![self.convert_type(other)],
                };
                let mut all_fields = vec![codepoints_array];
                all_fields.extend(capture_fields);
                Type::tuple_of(all_fields)
            }
        }
    }
}
