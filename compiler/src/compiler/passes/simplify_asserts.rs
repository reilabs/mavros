//! Splits complex expressions used as inputs to asserts into more-specialized chains of smaller
//! asserts that are far cheaper to encode in R1CS.
//!
//! Bare operations are specialized to `AssertCmp` (for `x < y` and `x == y`) and `AssertR1C` (for
//! `x == y`) wherever possible. Field equalities `assert(a * b == c)` also become native R1CS
//! constraints wherever possible.

use crate::{
    collections::HashMap,
    compiler::{
        analysis::{
            flow_analysis::FlowAnalysis,
            types::{FunctionTypeInfo, TypeInfo},
        },
        pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
        passes::shared::rewrite_asserts::rewrite_asserts,
        ssa::{
            ValueId,
            hlssa::{BinaryArithOpKind, CmpKind, HLSSA, OpCode, TypeExpr},
        },
    },
};

// SIMPLIFY ASSERTS
// ================================================================================================

pub struct SimplifyAsserts {}

impl Pass for SimplifyAsserts {
    fn name(&self) -> &'static str {
        "simplify_asserts"
    }
    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id()]
    }
    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>());
    }
    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id(), TypeInfo::id()]
    }
}

impl SimplifyAsserts {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo) {
        // Lower an equality via `emit_assert_eq` (the R1CS-native `AssertR1C` where a `Field`-mul
        // feeds it, else `AssertCmp{Eq}`). `lower_assert_cmp_eq = true` so an already-normalized
        // `AssertCmp{Eq}` instruction (e.g. emitted earlier by `NormalizeAsserts`) is lowered here
        // too — the witness-aware `Field`-mul → `AssertR1C` step lives only in this pass.
        rewrite_asserts(ssa, type_info, true, emit_assert_eq);
    }
}

// INTERNAL FUNCTIONS
// ================================================================================================

/// Lower an asserted equality `lhs == rhs` to the R1CS-native `AssertR1C` when a `Field` multiply
/// feeds either side, falling back to `AssertCmp{Eq}` otherwise.
///
/// This is the one piece `NormalizeAsserts` deliberately omits (it must stay witness-agnostic),
/// injected as [`rewrite_asserts`]'s equality handler.
fn emit_assert_eq(
    lhs: ValueId,
    rhs: ValueId,
    defs: &HashMap<ValueId, OpCode>,
    function_type_info: &FunctionTypeInfo,
) -> Vec<OpCode> {
    if let Some(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Mul,
        result,
        lhs: a,
        rhs: b,
    }) = defs.get(&lhs)
    {
        let result_type = function_type_info.get_value_type(*result);
        if matches!(result_type.strip_witness().expr, TypeExpr::Field) {
            return vec![OpCode::AssertR1C {
                a: *a,
                b: *b,
                c: rhs,
            }];
        }
    }

    if let Some(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Mul,
        result,
        lhs: a,
        rhs: b,
    }) = defs.get(&rhs)
    {
        let result_type = function_type_info.get_value_type(*result);
        if matches!(result_type.strip_witness().expr, TypeExpr::Field) {
            return vec![OpCode::AssertR1C {
                a: *a,
                b: *b,
                c: lhs,
            }];
        }
    }

    vec![OpCode::AssertCmp {
        kind: CmpKind::Eq,
        lhs,
        rhs,
    }]
}
