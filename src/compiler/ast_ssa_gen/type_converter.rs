//! Converts monomorphized AST types to mavros SSA types.

use noirc_frontend::monomorphization::ast::Type as AstType;
use noirc_frontend::shared::Signedness;

use crate::compiler::ir::r#type::{Empty, Type};

/// Converts AST types to SSA types.
pub struct AstTypeConverter;

impl AstTypeConverter {
    pub fn new() -> Self {
        Self
    }

    /// Convert a monomorphized AST type to an SSA type.
    pub fn convert_type(&self, ast_type: &AstType) -> Type<Empty> {
        match ast_type {
            AstType::Field => Type::field(Empty),
            AstType::Bool => Type::bool(Empty),
            AstType::Integer(signedness, bit_size) => {
                match signedness {
                    Signedness::Unsigned => Type::u(bit_size.bit_size() as usize, Empty),
                    Signedness::Signed => panic!("Signed integers not supported"),
                }
            }
            AstType::Unit => {
                // Unit type is represented as an empty tuple
                Type::tuple_of(vec![], Empty)
            }
            AstType::Array(len, elem_type) => {
                let elem = self.convert_type(elem_type);
                elem.array_of(*len as usize, Empty)
            }
            AstType::Slice(elem_type) => {
                let elem = self.convert_type(elem_type);
                elem.slice_of(Empty)
            }
            AstType::Tuple(types) => {
                let converted: Vec<Type<Empty>> = types
                    .iter()
                    .map(|t| self.convert_type(t))
                    .collect();
                Type::tuple_of(converted, Empty)
            }
            AstType::Reference(inner, _mutable) => {
                let inner_type = self.convert_type(inner);
                inner_type.ref_of(Empty)
            }
            AstType::Function(_, _, _, _) => Type::function(Empty),
            AstType::String(_) => todo!("String types not yet supported"),
            AstType::FmtString(_, _) => todo!("Format string types not yet supported"),
        }
    }
}
