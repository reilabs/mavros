//! Type conversion from Mavros types to LLVM types
//!
//! Currently only supports types needed for the `power` example:
//! - Field (BN254 field elements as [4 x i64])
//! - U (integers for loop counters)
//!
//! Uses WASM-friendly calling convention: all values passed directly.

use inkwell::context::Context;
use inkwell::types::BasicTypeEnum;

use crate::compiler::ir::r#type::{Type, TypeExpr};
use crate::compiler::taint_analysis::ConstantTaint;

/// Number of 64-bit limbs used to represent a BN254 field element
pub const FIELD_LIMBS: u32 = 4;

/// Converts Mavros types to LLVM types
pub struct TypeConverter<'ctx> {
    context: &'ctx Context,
}

impl<'ctx> TypeConverter<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        Self { context }
    }

    /// Convert a Mavros type to an LLVM type
    pub fn convert_type(&self, ty: &Type<ConstantTaint>) -> BasicTypeEnum<'ctx> {
        match &ty.expr {
            TypeExpr::Field => {
                // Field is represented as [4 x i64] - 256 bits in Montgomery form
                self.field_type().into()
            }
            TypeExpr::U(bits) => {
                // Map to appropriate LLVM integer type
                self.int_type(*bits).into()
            }
            _ => panic!("Unsupported type in LLVM codegen: {:?}", ty.expr),
        }
    }

    /// Get the LLVM type for a field element ([4 x i64])
    pub fn field_type(&self) -> inkwell::types::ArrayType<'ctx> {
        self.context.i64_type().array_type(FIELD_LIMBS)
    }

    /// Get an integer type of the specified bit width
    pub fn int_type(&self, bits: usize) -> inkwell::types::IntType<'ctx> {
        match bits {
            1 => self.context.bool_type(),
            8 => self.context.i8_type(),
            16 => self.context.i16_type(),
            32 => self.context.i32_type(),
            64 => self.context.i64_type(),
            _ => self.context.custom_width_int_type(bits as u32),
        }
    }
}
