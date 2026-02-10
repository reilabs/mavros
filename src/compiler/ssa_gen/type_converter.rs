use noirc_evaluator::ssa::ir::types::{NumericType, Type as NoirType};

use crate::compiler::ir::r#type::{Empty, Type};

pub struct TypeConverter;

impl TypeConverter {
    pub fn new() -> Self {
        TypeConverter
    }

    pub fn convert_type(&self, noir_type: &NoirType) -> Type<Empty> {
        match noir_type {
            NoirType::Numeric(numeric) => match numeric {
                NumericType::NativeField => Type::field(Empty),
                NumericType::Unsigned { bit_size: 1 } => Type::u(1, Empty),
                NumericType::Unsigned { bit_size } => {
                    Type::u((*bit_size).try_into().unwrap(), Empty)
                }
                NumericType::Signed { bit_size } => {
                    panic!("Signed integers not supported: {} bits", bit_size)
                }
            },
            NoirType::Reference(inner) => {
                let inner_converted = self.convert_type(inner);
                Type::ref_of(inner_converted, Empty)
            }
            NoirType::Array(element_types, size) => {
                let tp = if element_types.len() == 1 {
                    let element_type = &element_types[0];
                    let converted_element = self.convert_type(element_type);
                    Type::array_of(converted_element, (*size).try_into().unwrap(), Empty)
                } else {
                    let converted_types: Vec<Type<Empty>> = element_types.iter()
                        .map(|t| self.convert_type(t))
                        .collect();
                    Type::array_of(Type::tuple_of(converted_types, Empty), (*size).try_into().unwrap(), Empty)
                };
                tp
            }
            NoirType::Slice(element_types) => {
                if element_types.len() != 1 {
                    panic!(
                        "Only single-type slices are supported, got element_types = {:?}",
                        element_types
                    )
                }
                let element_type = &element_types[0];
                let converted_element = self.convert_type(element_type);
                Type::slice_of(converted_element, Empty)
            }
            NoirType::Function => Type::function(Empty),
        }
    }
}
