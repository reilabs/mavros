//! Witness-agnostic assert normalization that never produces the R1CS-native `AssertR1C`, so it is
//! safe to run early in the pipeline (including before witness typing).
//!
//! It specializes `Assert{v=Cmp{Eq,a,b}} → AssertCmp{Eq,a,b}`, `Assert{v=Cmp{Lt,..}} →
//! AssertCmp{Lt,..}`, and splits a `u1`-`And` assert into one assert per operand. Turning
//! `Assert{Cmp{Eq}}` into `AssertCmp{Eq}` here is what makes the conditional facts (asserted
//! equalities / asserted constants) available to the early SCS sites, on still-scalar operands.
//!
//! The `Field`-mul → `AssertR1C` lowering is deliberately left to
//! [`SimplifyAsserts`](super::simplify_asserts::SimplifyAsserts), which must run late (after
//! witness typing / lowering).

use crate::compiler::{
    analysis::{flow_analysis::FlowAnalysis, types::TypeInfo},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    passes::shared::rewrite_asserts::rewrite_asserts,
    ssa::hlssa::{CmpKind, HLSSA, OpCode},
};

// ASSERT NORMALIZATION
// ================================================================================================

pub struct NormalizeAsserts {}

impl Pass for NormalizeAsserts {
    fn name(&self) -> &'static str {
        "normalize_asserts"
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

impl NormalizeAsserts {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo) {
        // Lower an equality to the witness-agnostic `AssertCmp{Eq}`
        rewrite_asserts(ssa, type_info, false, |lhs, rhs, _defs, _fti| {
            vec![OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs,
                rhs,
            }]
        });
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        analysis::types::Types,
        ssa::{
            Terminator, ValueId,
            hlssa::{BinaryArithOpKind, Type},
        },
    };

    /// Run only the witness-agnostic normalization (no `SimplifyAsserts`, no DCE).
    fn normalize(ssa: &mut HLSSA) {
        let flow = FlowAnalysis::run(ssa);
        let type_info = Types::new().run(ssa, &flow);
        NormalizeAsserts::new().do_run(ssa, &type_info);
    }

    /// AssertCmp ops as `(is_eq, lhs, rhs)`. `CmpKind` has no `PartialEq`, so the kind is flattened
    /// to a bool (true = Eq, false = Lt) for easy vector comparison.
    fn assert_cmps(ssa: &HLSSA) -> Vec<(bool, ValueId, ValueId)> {
        ssa.get_unique_entrypoint()
            .get_entry()
            .get_instructions()
            .filter_map(|i| match i {
                OpCode::AssertCmp { kind, lhs, rhs } => {
                    Some((matches!(kind, CmpKind::Eq), *lhs, *rhs))
                }
                _ => None,
            })
            .collect()
    }

    fn has_r1c(ssa: &HLSSA) -> bool {
        ssa.get_unique_entrypoint()
            .get_entry()
            .get_instructions()
            .any(|i| matches!(i, OpCode::AssertR1C { .. }))
    }

    /// `assert(a == b)` normalizes to `AssertCmp{Eq, a, b}`, and — unlike `SimplifyAsserts` — never
    /// to the R1CS-native `AssertR1C`, even for a `Field`-mul left-hand side.
    #[test]
    fn assert_of_eq_cmp_becomes_assert_cmp_eq_without_r1c() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (a, b, mul, c, eq) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(a, Type::field());
        f.get_entry_mut().push_parameter(b, Type::field());
        f.get_entry_mut().push_parameter(c, Type::field());
        let entry = f.get_entry_mut();
        // A `Field` multiply feeding the equality: `SimplifyAsserts` would turn this into `AssertR1C`,
        // but `NormalizeAsserts` must not.
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Mul,
            result: mul,
            lhs: a,
            rhs: b,
        });
        entry.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: mul,
            rhs: c,
        });
        entry.push_instruction(OpCode::Assert { value: eq });
        entry.set_terminator(Terminator::Return(vec![]));

        normalize(&mut ssa);

        assert_eq!(assert_cmps(&ssa), vec![(true, mul, c)]);
        assert!(!has_r1c(&ssa), "NormalizeAsserts must not emit AssertR1C");
    }

    /// `assert(a < b)` normalizes to `AssertCmp{Lt, a, b}`.
    #[test]
    fn assert_of_lt_cmp_becomes_assert_cmp_lt() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (a, b, lt) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(a, Type::field());
        f.get_entry_mut().push_parameter(b, Type::field());
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::Cmp {
            kind: CmpKind::Lt,
            result: lt,
            lhs: a,
            rhs: b,
        });
        entry.push_instruction(OpCode::Assert { value: lt });
        entry.set_terminator(Terminator::Return(vec![]));

        normalize(&mut ssa);

        assert_eq!(assert_cmps(&ssa), vec![(false, a, b)]);
    }

    /// `assert(l & r)` over `u1` splits into `assert(l)`, `assert(r)`.
    #[test]
    fn assert_of_u1_and_splits_into_two_asserts() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (l, r, and) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(l, Type::u(1));
        f.get_entry_mut().push_parameter(r, Type::u(1));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::And,
            result: and,
            lhs: l,
            rhs: r,
        });
        entry.push_instruction(OpCode::Assert { value: and });
        entry.set_terminator(Terminator::Return(vec![]));

        normalize(&mut ssa);

        let asserts: Vec<ValueId> = ssa
            .get_unique_entrypoint()
            .get_entry()
            .get_instructions()
            .filter_map(|i| match i {
                OpCode::Assert { value } => Some(*value),
                _ => None,
            })
            .collect();
        assert_eq!(asserts, vec![l, r]);
    }

    /// An asserted boolean that is neither a comparison nor a `u1`-`And` is left as a bare `Assert`.
    #[test]
    fn opaque_assert_is_left_unchanged() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let w = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(w, Type::u(1));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::Assert { value: w });
        entry.set_terminator(Terminator::Return(vec![]));

        normalize(&mut ssa);

        assert!(assert_cmps(&ssa).is_empty());
        let asserts: Vec<ValueId> = ssa
            .get_unique_entrypoint()
            .get_entry()
            .get_instructions()
            .filter_map(|i| match i {
                OpCode::Assert { value } => Some(*value),
                _ => None,
            })
            .collect();
        assert_eq!(asserts, vec![w]);
    }
}
