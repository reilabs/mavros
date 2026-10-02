//! Converts monomorphized AST types to mavros SSA types.

use noirc_frontend::monomorphization::ast::Type as NoirType;
use noirc_frontend::shared::Signedness;

use mavros_int_semantics::MAX_LOWERED_SIGNED_BITS;

use crate::compiler::ssa::hlssa::Type;

/// Converts AST types to SSA types.
pub struct TypeConverter;

impl TypeConverter {
    pub fn new() -> Self {
        Self
    }

    /// Convert a monomorphized AST type to an SSA type.
    pub fn convert_type(&self, ast_type: &NoirType) -> Type {
        match ast_type {
            NoirType::Field => Type::field(),
            NoirType::Bool => Type::bool(),
            NoirType::Integer(signedness, bit_size) => match signedness {
                Signedness::Unsigned => Type::int(bit_size.bit_size() as usize),
                Signedness::Signed => {
                    let bits = bit_size.bit_size() as usize;
                    assert!(
                        bits <= MAX_LOWERED_SIGNED_BITS,
                        "signed integers wider than i{MAX_LOWERED_SIGNED_BITS} are unsupported"
                    );
                    Type::int(bits)
                }
            },
            NoirType::Unit => {
                // Unit type is represented as an empty tuple
                Type::tuple_of(vec![])
            }
            NoirType::Array(len, elem_type) => {
                let elem = self.convert_type(elem_type);
                elem.array_of(*len as usize)
            }
            NoirType::Vector(elem_type) => {
                let elem = self.convert_type(elem_type);
                elem.slice_of()
            }
            NoirType::Tuple(types) => {
                let converted: Vec<Type> = types.iter().map(|t| self.convert_type(t)).collect();
                Type::tuple_of(converted)
            }
            NoirType::Reference(inner, _mutable) => {
                let inner_type = self.convert_type(inner);
                inner_type.ref_of()
            }
            // Only the results survive. The parameter list is dropped because a closure's lifted
            // function takes its captured environment as an extra leading parameter, so `args` is
            // not the arity of the call this value appears in -- see [`TypeExpr::Function`]. The
            // results are the same either way, and they are what lets an indirect call be typed
            // before defunctionalization has run.
            NoirType::Function(_, ret, _, _) => {
                Type::function_returning(vec![self.convert_type(ret)])
            }
            NoirType::String(len) => {
                // str<N>: N is UTF-8 byte count, represented as Array(U(8), N)
                Type::int(8).array_of(*len as usize)
            }
            NoirType::FmtString(len, captures_type) => {
                // fmtstr<N, T>: N is codepoint count, represented as
                // Tuple(Array(U(32), N), ...T_fields)
                let codepoints_array = Type::int(32).array_of(*len as usize);
                let capture_fields = match captures_type.as_ref() {
                    NoirType::Tuple(fields) => fields
                        .iter()
                        .map(|t| self.convert_type(t))
                        .collect::<Vec<_>>(),
                    NoirType::Unit => vec![],
                    other => vec![self.convert_type(other)],
                };
                let mut all_fields = vec![codepoints_array];
                all_fields.extend(capture_fields);
                Type::tuple_of(all_fields)
            }
        }
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use std::rc::Rc;

    fn function_type(ret: NoirType) -> NoirType {
        NoirType::Function(
            vec![NoirType::Field],
            Rc::new(ret),
            Rc::new(NoirType::Unit),
            false,
        )
    }

    #[test]
    fn a_function_type_converts_to_its_results() {
        let converter = TypeConverter::new();

        assert_eq!(
            converter.convert_type(&function_type(NoirType::Field)),
            Type::function_returning(vec![Type::field()])
        );
        assert_eq!(
            converter.convert_type(&function_type(NoirType::Unit)),
            Type::function_returning(vec![Type::tuple_of(vec![])])
        );
    }

    #[test]
    fn two_function_types_differing_only_in_their_parameters_convert_alike() {
        let converter = TypeConverter::new();
        let no_arguments = NoirType::Function(
            vec![],
            Rc::new(NoirType::Field),
            Rc::new(NoirType::Unit),
            false,
        );

        assert_eq!(
            converter.convert_type(&function_type(NoirType::Field)),
            converter.convert_type(&no_arguments)
        );
    }
}
