//! Totality classification for speculative placement.
//!
//! PRE may evaluate an op at a point where the original program was not bound to evaluate it
//! (speculation — e.g. hoisting a loop-body computation above a zero-trip-able loop header) only
//! if the op is _total at the emplacement point_: on every run reaching that point it can neither
//! trap any engine (witgen VM, LLVM/WASM, the R1CS generator's constant evaluation) nor introduce
//! a constraint able to reject. Otherwise a run the original program accepted could come to reject
//! — or stop compiling — which breaks obligation 1 of the accept/reject model (see `click_cooper`'s
//! "Soundness on Rejecting Runs").
//!
//! Down-safe motion needs no such gate: an op bound to execute on every path from the insertion
//! point merely traps _earlier_ on runs already doomed to reject, and behaves identically on
//! accepting runs. This module is consulted exclusively for placements that are **not** down-safe.
//!
//! The verdicts encode the engines' actual semantics:
//!
//! - **U/I `Add`/`Sub`/`Mul` are Never Speculated** Even though the executable backends wrap
//!   (`vm/src/bytecode.rs` `cell_add`/`add_int128` mask to width): overflow is Noir-semantically an
//!   error — the constant lattice refuses overflow folds as "an erroneous evaluation with a
//!   backend-specific residue" — and a _witness-typed_ op's gadget lowering emits a rejecting
//!   overflow `Rangecheck` when the value-range analysis cannot prove fit. Field arithmetic has no
//!   such predicate and is total.
//! - **`Div`/`Mod` are Undefined on a Zero Divisor**: LLVM's `udiv`/`sdiv` make it poison and the
//!   VM's `cell_udiv` and friends answer zero — an arbitrary choice made so the interpreter stays
//!   total, not an actual answer. `mavros-int-semantics` records that disagreement by having
//!   `residue` return `None` for exactly these inputs. Meanwhile _field_ division is total in the
//!   VM (`div_field` yields 0) but is rejected by the R1CS generator's constant evaluation, so both
//!   domains are gated. This is about the _opcode_ since every `Div`/`Mod` also carries a preceding
//!   assert that the operands are defined, so a bad division fails the program even where the
//!   opcode itself would not. That assert is a separate instruction this oracle never speculates,
//!   which is why the gating below is still exactly what keeps a division and its check together.
//!   Divisions are speculated only where the divisor is provably nonzero: a nonzero constant, or
//!   the analysis's disequality channel ([`ClickCooper::known_unequal`] against the interned zero
//!   of the divisor's type) at the insertion block. A **signed** division additionally owes the
//!   `INT_MIN / -1` overflow rejection that `divmod_guard::emit_divmod_failure_cond` emits at every
//!   width, so it is speculated only where the divisor is a constant other than `-1`.
//! - **`Shl`/`Shr` Reject an Out-of-Range Amount:** An amount at or past the operand's width is a
//!   constraint failure in Noir so `mavros-int-semantics`' `eval` reports `Reject::ShiftAmount` and
//!   speculating one would turn an accepting run into a rejecting one. PRE thus only speculates on
//!   a constant, in-range amount.
//! - **`ArrayGet`/`ArraySet` Trap on OOB:** VM asserts, R1CGen constant evaluation panics; only a
//!   constant index provably below a static array length is speculated.
//! - **Comparisons, `Not`, `Select`, `SExt`, `BitRange`, bitwise ops, and Casts are Total:** Their
//!   witness gadget lowerings emit only _functional_ decomposition constraints (satisfiable for
//!   every in-range input), never a semantic predicate.
//!
//! The oracle answers for individual ops; whether an op is a motion candidate at all (the
//! CSE-parity op filter, witness-machinery exclusions, aggregate deferral) is the caller's concern
//! — verdicts here are honest for every op, including ones PRE never moves, with one deliberate
//! exception: the element-wise `Map` cast does not trap but is answered `false` anyway (it is never
//! worth materializing at a new point, and the motion candidate filter excludes it too).

use mavros_artifacts::FieldConfig;

use crate::compiler::{
    analysis::{
        click_cooper::ClickCooper, types::FunctionTypeInfo,
        witness_taint_inference::ApproximateWitnessTaint,
    },
    ssa::{
        BlockId, FunctionId, ValueId,
        hlssa::{
            ArithGroup, BinaryArithOpKind, CastTarget, Constant, HLSSA, OpCode, Type, TypeExpr,
        },
    },
};

// TOTALITY ORACLE
// ================================================================================================

/// Answers "is `op` total on entry to a given block?" against the Click–Cooper facts of one
/// function.
pub struct TotalityOracle<'a> {
    /// The analysis facts the verdicts are conditioned on: `const_of` discharges the
    /// constant-divisor / shift-amount / array-index gates, and `known_unequal` (the disequality
    /// channel) discharges a non-constant divisor at blocks covered by a `!= 0` branch.
    cc: &'a ClickCooper,

    /// The whole program, consulted only for its interned-constants table: resolving the zero of
    /// the divisor's type to the `ValueId` a disequality fact would be keyed against
    /// ([`SSA::find_const`](crate::compiler::ssa::SSA::find_const)).
    ssa: &'a HLSSA,

    /// The function whose facts in [`Self::cc`] are queried.
    ///
    /// All ops and blocks passed to [`Self::is_total_at`] must belong to it.
    fid: FunctionId,

    /// `fid`'s value types, used to pick the semantic domain of a verdict (Field vs. U/I, bit
    /// widths, static array lengths) and to refuse witness-typed shapes. Must be the _pristine_
    /// typing of the function being planned over: [`FunctionTypeInfo::get_value_type`] panics on
    /// ids it has not seen, so no queried op may mention a value minted after this was computed.
    types: &'a FunctionTypeInfo,

    /// Where the four witness-ness-gated verdicts read witness-ness from (see
    /// [`WitnessnessSource`]). Post-untaint callers use [`Self::new`] (`Types`); the pre-untaint
    /// caller must supply `Taint` via [`Self::with_witness_source`].
    witness: WitnessnessSource<'a>,
}

impl<'a> TotalityOracle<'a> {
    /// An oracle for `fid`, answering against the `cc` facts and `types` typing computed over the
    /// same, unmutated version of the function (see the field docs for what each input supplies).
    ///
    /// Witness-ness is read from the types, so this constructor is only sound **post-untaint**.
    pub fn new(
        cc: &'a ClickCooper,
        ssa: &'a HLSSA,
        fid: FunctionId,
        types: &'a FunctionTypeInfo,
    ) -> Self {
        Self::with_witness_source(cc, ssa, fid, types, WitnessnessSource::Types)
    }

    /// [`Self::new`] with an explicit witness-ness source, for the pre-untaint caller. In `Types`
    /// mode this is exactly [`Self::new`].
    pub fn with_witness_source(
        cc: &'a ClickCooper,
        ssa: &'a HLSSA,
        fid: FunctionId,
        types: &'a FunctionTypeInfo,
        witness: WitnessnessSource<'a>,
    ) -> Self {
        Self {
            cc,
            ssa,
            fid,
            types,
            witness,
        }
    }

    /// `true` if evaluating `op` on entry to `block` can neither trap nor emit a rejecting
    /// constraint on _any_ run reaching `block` — the license to place it at a point where it was
    /// not bound to execute.
    ///
    /// Every queried operand must be known to the pass's input `TypeInfo` (the caller plans against
    /// the pristine function, so this holds by construction).
    pub fn is_total_at(&self, op: &OpCode, block: BlockId) -> bool {
        use ArithGroup::*;
        match op {
            OpCode::BinaryArithOp { kind, lhs, rhs, .. } => match kind.group() {
                And | Or | Xor => true,
                // Field arithmetic (witness-typed or not) is total; U/I overflow is an error.
                Add | Sub | Mul => self.value_type(*lhs).peel_witness().is_field(),
                // A division with _any_ witness-typed operand lowers to a constraint-emitting
                // gadget whose guarded and unguarded shapes differ; never speculated.
                Div | Rem => {
                    !self.is_witness(*lhs) && self.divisor_provably_safe(*kind, *rhs, block)
                }
                Shl | Shr => self.shift_amount_in_range(*lhs, *rhs),
            },
            // Multiplication by an interned constant: the same overflow story as `Mul`.
            OpCode::MulConst { var, .. } => self.value_type(*var).peel_witness().is_field(),
            OpCode::Cmp { .. }
            | OpCode::Not { .. }
            | OpCode::Select { .. }
            | OpCode::SExt { .. }
            | OpCode::BitRange { .. } => true,
            OpCode::Cast { target, .. } => match target {
                // Raw-bits / representation-only conversions; narrowing truncates in every
                // engine (`cast_field_to_u64` takes the low limb), it does not trap.
                CastTarget::Nop
                | CastTarget::Field
                | CastTarget::Int(_)
                | CastTarget::WitnessOf
                | CastTarget::ValueOf
                | CastTarget::ArrayToSlice => true,
                // Lowered into an element-wise loop late; never worth speculating.
                CastTarget::Map(_) => false,
            },
            // Pure value-semantic sequence constructors never trap.
            OpCode::MkSeq { .. }
            | OpCode::MkRepeated { .. }
            | OpCode::MkSeqOfBlob { .. }
            | OpCode::SlicePush { .. }
            | OpCode::SliceLen { .. } => true,
            OpCode::ArrayGet { array, index, .. } | OpCode::ArraySet { array, index, .. } => {
                self.const_index_in_bounds(*array, *index)
            }
            // Everything else — asserts, constraint emitters, witness machinery, memory, calls,
            // globals, bit decompositions, guards — is effectful or constraint-bearing and never
            // total at a speculative placement.
            _ => false,
        }
    }

    /// The type of `v` in the oracle's function. Panics if `v` is unknown to [`Self::types`] (a
    /// value minted after the pass's input typing was computed).
    fn value_type(&self, v: ValueId) -> &Type {
        self.types.get_value_type(v)
    }

    /// Whether `v` is witness at its top level, read from the configured [`WitnessnessSource`].
    ///
    /// In `Types` mode this is the exact pre-existing check, `Type::is_witness_of`.
    fn is_witness(&self, v: ValueId) -> bool {
        match &self.witness {
            WitnessnessSource::Types => self.value_type(v).is_witness_of(),
            WitnessnessSource::Taint(taint) => taint.value_is_witness(self.fid, v),
        }
    }

    /// The divisor of a `Div`/`Mod` is provably safe at `block`: nonzero via a constant or the
    /// disequality channel, and free of the `i64::MIN / -1` overflow.
    fn divisor_provably_safe(
        &self,
        kind: BinaryArithOpKind,
        divisor: ValueId,
        block: BlockId,
    ) -> bool {
        if self.is_witness(divisor) {
            return false;
        }
        let field = self.ssa.field();
        let ty = self.value_type(divisor);

        // `INT_MIN / -1` is a failure at **every** signed width, not just 64. What the operation
        // owes is set by `divmod_guard::emit_divmod_failure_cond`, which builds the overflow
        // disjunct from `1 << (bits - 1)` for any `bits` — so an `i8` division by a constant `-1`
        // can still fail, and speculating it moves that failure onto a path that did not have it.
        //
        // This used to be gated on `bits == 64`, reasoning from the VM instead: `sdiv_int`
        // sign-extends to i64, where only a 64-bit `MIN / -1` actually overflows. That is a true
        // statement about `sdiv_int` and the wrong question — the check the speculation has to
        // respect is the one mavros emits, not the one the interpreter would trap on.
        let minus_one_hazard = kind.is_signed();

        if let Some(c) = self.cc.const_of(self.fid, divisor) {
            return !constant_is_zero(&c, field) && !(minus_one_hazard && constant_is_all_ones(&c));
        }
        if minus_one_hazard {
            return false;
        }

        // Non-constant divisor: the disequality channel. A branch fact against zero can only exist
        // if the zero of the divisor's type is already interned.
        let Some(zero) = zero_constant_of(ty, field) else {
            return false;
        };
        let Some(zero_id) = self.ssa.find_const(&zero) else {
            return false;
        };
        self.cc.known_unequal(self.fid, block, divisor, zero_id)
    }

    /// The shift amount is a constant strictly below the shifted operand's width (the range the
    /// constant lattice folds and every backend agrees on).
    fn shift_amount_in_range(&self, lhs: ValueId, rhs: ValueId) -> bool {
        // Witness-typed shifts lower to decomposition gadgets; never speculated (see the pass
        // module doc's Deferred Improvements).
        if self.is_witness(lhs) || self.is_witness(rhs) {
            return false;
        }
        let lhs_ty = self.value_type(lhs);
        let Some(c) = self.cc.const_of(self.fid, rhs) else {
            return false;
        };
        let Some(amount) = constant_as_u128(&c) else {
            return false;
        };
        amount < lhs_ty.get_bit_size(self.ssa.field()) as u128
    }

    /// The index is a constant strictly below a _static_ array length.
    ///
    /// Slices have no static length and a witness-typed *index* lowers to a constraint-emitting
    /// gadget — both refused. The array itself is never consulted for witness-ness: container
    /// tops carry no taint, so it cannot be witness-typed.
    fn const_index_in_bounds(&self, array: ValueId, index: ValueId) -> bool {
        if self.is_witness(index) {
            return false;
        }
        let arr_ty = self.value_type(array);
        let TypeExpr::Array(_, len) = &arr_ty.expr else {
            return false;
        };
        let Some(c) = self.cc.const_of(self.fid, index) else {
            return false;
        };
        let Some(i) = constant_as_u128(&c) else {
            return false;
        };
        i < *len as u128
    }
}

// WITNESS-NESS SOURCE
// ================================================================================================

/// Where the oracle reads a value's witness-ness from.
///
/// Four verdicts gate on witness-ness because the op's _witness_ lowering emits rejecting
/// constraints its pure lowering does not (`Div`/`Mod` gadgets, shift decompositions, array-access
/// gadgets). Post-untaint that is a type property (`TypeExpr::WitnessOf`); pre-untaint those types
/// do not exist yet, and reading them would silently answer "pure" for every value — licensing
/// unsound speculation. A pre-untaint caller must instead supply the taint approximation
/// ([`ApproximateWitnessTaint`]), which answers the same question about every _future_
/// specialization.
pub enum WitnessnessSource<'a> {
    /// Post-untaint: witness-ness is baked into the types the oracle already holds.
    Types,

    /// Pre-untaint: witness-ness from the read-only joined WTI solve over the pristine SSA.
    Taint(&'a ApproximateWitnessTaint),
}

// INTERNAL UTILITIES
// ================================================================================================

/// The zero constant of a scalar type, or `None` for non-scalar types.
fn zero_constant_of(ty: &Type, field: FieldConfig) -> Option<Constant> {
    match &ty.expr {
        TypeExpr::Int(bits) => Some(Constant::Int(*bits, 0)),
        TypeExpr::Field => Some(Constant::Field(field.zero())),
        _ => None,
    }
}

/// `true` if `c` is the zero of its scalar domain (`FnPtr`/`Blob` constants have none).
fn constant_is_zero(c: &Constant, field: FieldConfig) -> bool {
    match c {
        Constant::Int(_, v) => *v == 0,
        Constant::Field(f) => *f == field.zero(),
        Constant::FnPtr(_) | Constant::Blob(_) => false,
    }
}

/// `true` if `c` is the all-ones bit pattern of its width — the two's-complement `-1` for a
/// signed constant.
fn constant_is_all_ones(c: &Constant) -> bool {
    match c {
        Constant::Int(bits, v) => {
            let mask = if *bits >= 128 {
                u128::MAX
            } else {
                (1u128 << bits) - 1
            };
            *v & mask == mask
        }
        Constant::Field(_) | Constant::FnPtr(_) | Constant::Blob(_) => false,
    }
}

/// The raw bit pattern of an integer constant, or `None` for the non-integer variants.
fn constant_as_u128(c: &Constant) -> Option<u128> {
    match c {
        Constant::Int(_, v) => Some(*v),
        Constant::Field(_) | Constant::FnPtr(_) | Constant::Blob(_) => None,
    }
}

/// The totality oracle's conformance relation to the normative model in `mavros-int-semantics`.
///
/// [`TotalityOracle::is_total_at`] answers a question the model also answers from the other side:
/// the oracle asks whether an operation is safe to evaluate where it was not bound to, and the
/// model says which operations can be _rejected_ at all. So the relation is
///
/// > `is_total_at` may answer an unconditional `true` for an [`ArithGroup`] **iff** no input makes
/// > [`eval`](mavros_int_semantics::eval) reject it.
///
/// One direction keeps the pass sound: speculating an operation that can reject turns an accepting
/// run into a rejecting one, which is obligation 1 of the accept/reject model. The other keeps it
/// useful: gating an operation that can never fail costs optimisation for nothing.
///
/// The claim table below is written out rather than read off the oracle, because reaching
/// `is_total_at` needs a whole analysis context and the answer would then be about one program
/// rather than about the group. The value is in the table being **exhaustive**: adding an
/// `ArithGroup` fails to compile here, and changing the model's rejection set fails the assertion.
#[cfg(test)]
mod int_semantics_conformance {
    use mavros_int_semantics::{IntOp, Sign, corners, eval};

    use super::*;

    /// What `is_total_at`'s `BinaryArithOp` arm claims about an operation on **integers**.
    #[derive(Debug, PartialEq, Eq)]
    enum Claim {
        /// Speculated freely, with no predicate consulted.
        Unconditional,

        /// Speculated only where some further condition is discharged: the operand being field
        /// typed, the divisor provably safe, the shift amount provably in range.
        Conditional,
    }

    /// The claim the oracle makes, transcribed from its `match` arms one for one.
    fn claim(group: ArithGroup) -> Claim {
        match group {
            // `And | Or | Xor => true`.
            ArithGroup::And | ArithGroup::Or | ArithGroup::Xor => Claim::Unconditional,
            // `Add | Sub | Mul => ..is_field()`, which is false for every integer.
            ArithGroup::Add | ArithGroup::Sub | ArithGroup::Mul => Claim::Conditional,
            // `Div | Rem => !is_witness(lhs) && divisor_provably_safe(..)`.
            ArithGroup::Div | ArithGroup::Rem => Claim::Conditional,
            // `Shl | Shr => shift_amount_in_range(..)`.
            ArithGroup::Shl | ArithGroup::Shr => Claim::Conditional,
        }
    }

    /// Whether the model rejects this operation on any corner input at any width.
    fn can_reject(group: ArithGroup) -> bool {
        let op: IntOp = group.into();

        for sign in Sign::ALL {
            for &bits in corners::widths_for(sign.is_signed()) {
                let rhs = if matches!(op, IntOp::Shl | IntOp::Shr) {
                    corners::shift_amounts(bits, bits)
                } else {
                    corners::values(bits)
                };

                for a in corners::values(bits) {
                    for b in &rhs {
                        if eval(op, sign, bits, a, bits, *b).is_rejected() {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    #[test]
    fn an_unconditional_verdict_is_given_exactly_where_nothing_can_reject() {
        for group in [
            ArithGroup::Add,
            ArithGroup::Sub,
            ArithGroup::Mul,
            ArithGroup::Div,
            ArithGroup::Rem,
            ArithGroup::Shl,
            ArithGroup::Shr,
            ArithGroup::And,
            ArithGroup::Or,
            ArithGroup::Xor,
        ] {
            let expected = if can_reject(group) {
                Claim::Conditional
            } else {
                Claim::Unconditional
            };

            assert_eq!(
                claim(group),
                expected,
                "{group:?}: the oracle's verdict and the model's rejection set disagree"
            );
        }
    }

    #[test]
    fn the_bitwise_operations_are_the_only_total_ones() {
        // The same fact stated positively, so that a model change which made, say, `Shl` total
        // reads as a deliberate edit here rather than as a mysterious flip above.
        let total: Vec<_> = [
            ArithGroup::Add,
            ArithGroup::Sub,
            ArithGroup::Mul,
            ArithGroup::Div,
            ArithGroup::Rem,
            ArithGroup::Shl,
            ArithGroup::Shr,
            ArithGroup::And,
            ArithGroup::Or,
            ArithGroup::Xor,
        ]
        .into_iter()
        .filter(|g| !can_reject(*g))
        .collect();

        assert_eq!(
            total,
            vec![ArithGroup::And, ArithGroup::Or, ArithGroup::Xor],
            "the set of operations no input can reject has changed"
        );
    }
}
