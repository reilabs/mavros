//! Creates specialized copies of functions for specific call-site argument values using symbolic
//! execution where beneficial.
//!
//! These specialized functions are wired in using a dispatch function mechanism similar to the one
//! in defunctionalization wherever needed. In some cases, we can call the specialized version
//! directly instead.

use ark_ff::BigInteger;
use tracing::{info, instrument};

use mavros_artifacts::FieldConfig;

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        Field,
        analysis::{
            click_cooper::lattice,
            instrumenter::{FunctionSignature, SpecializationSummary, Summary, ValueSignature},
            symbolic_executor::{self, AssertionFailure, SymbolicExecutor},
            types::TypeInfo,
        },
        pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
        ssa::{
            BlockId, FunctionId, SourceLocation, ValueId,
            hlssa::{
                ArithGroup, BinaryArithOpKind, Blob, CastTarget, CmpKind, Constant, Endianness,
                HLFunction, HLSSA, LocatedOpCode, LookupTarget, MAX_SUPPORTED_UNSIGNED_BITS,
                OpCode, Radix, RefCountOp, SequenceTargetType, Type,
                builder::{HLEmitter, HLFunctionBuilder},
            },
        },
        util::{spread_bits, unspread_bits},
    },
};

/// Whether this pass folds `group` on a constant pair whose left side is `lhs`, or leaves it to
/// `ClickCooper`.
///
/// Everything except a **field division**. `lattice::eval_binary` does fold one by a nonzero
/// constant divisor — but minting that constant _here_ would move the cost estimate this pass takes
/// its own specialization decisions on, which is a change to what gets specialized rather than the
/// meaning of an operation. `ClickCooper` folds it either way, after those decisions are made.
fn folds_here(group: ArithGroup, lhs: &Constant) -> bool {
    !(group == ArithGroup::Div && matches!(lhs, Constant::Field(_)))
}

/// The scalar [`Constant`] a `ConstVal` denotes, for handing to the shared lattice folders.
///
/// Aggregates answer `None`: the folders are scalar, and an `Array`/`Blob`/`BitsOf` holds
/// `ValueId`s rather than a value this pass can evaluate.
fn const_val_scalar(value: Option<&ConstVal>) -> Option<Constant> {
    match value? {
        ConstVal::Int(s, v) => Some(Constant::Int(*s, *v)),
        ConstVal::Field(f) => Some(Constant::Field(*f)),
        ConstVal::Array(_) | ConstVal::Blob(_) | ConstVal::BitsOf(..) => None,
    }
}

/// Assert that a recorded integer constant is the width the operation says its operands are.
///
/// A no-op for a `Field` (which has no width to disagree about) and for a value with no recorded
/// constant, and for an operation that did not name a width. See [`Val::cmp`] for why the two
/// widths agreeing is what lets the fold read one off the constant.
fn assert_recorded_width(bits: Option<usize>, recorded: Option<&Constant>, kind: CmpKind) {
    if let (Some(bits), Some(Constant::Int(s, v))) = (bits, recorded) {
        assert_eq!(
            *s, bits,
            "{kind:?} on a {bits}-bit operand whose recorded constant {v:#x} is {s}-bit: the two \
             readings of that pattern differ"
        );
    }
}

/// Intern a folded scalar constant and record its value, so later folds in the same run see it.
///
/// The specializer folds a whole callee body against one call site's arguments, so a constant it
/// mints here is an input to the next instruction it walks.
fn intern_folded(ctx: &mut SpecializationState, folded: Constant) -> Option<Val> {
    match folded {
        Constant::Int(s, v) => {
            let id = ctx.int_const(s, v);
            ctx.const_vals.insert(id, ConstVal::Int(s, v));
            Some(Val(id))
        }
        Constant::Field(f) => {
            let id = ctx.field_const(f);
            ctx.const_vals.insert(id, ConstVal::Field(f));
            Some(Val(id))
        }
        Constant::FnPtr(_) | Constant::Blob(_) => None,
    }
}

pub struct Specializer {
    pub savings_to_code_ratio: f64,
}

#[derive(Debug, Clone)]
// FIELD-ASSUMPTION: L4-eval
enum ConstVal {
    Int(usize, u128),
    Field(Field),
    Array(Vec<ValueId>),
    Blob(Vec<ValueId>),
    BitsOf(Box<ValueId>, usize, Endianness),
}

/// The width and raw payload of an integer `ConstVal`.
///
/// Consumers that care about signedness decode with the width themselves, using the sign the
/// _operation_ names. Returns `None` for non-integer constants, which never fold.
fn int_bits_and_raw(value: Option<&ConstVal>) -> Option<(usize, u128)> {
    match value? {
        ConstVal::Int(s, v) => Some((*s, *v)),
        ConstVal::Field(_) | ConstVal::Array(_) | ConstVal::Blob(_) | ConstVal::BitsOf(..) => None,
    }
}

fn const_val_as_field(value: &ConstVal, field: FieldConfig) -> Option<Field> {
    match value {
        ConstVal::Int(_, v) => Some(field.constant(*v)),
        ConstVal::Field(f) => Some(*f),
        _ => None,
    }
}

/// The operands of an `a * b = c` constraint as field constants, if all three are known.
fn r1c_consts(
    ctx: &SpecializationState<'_>,
    a: &Val,
    b: &Val,
    c: &Val,
) -> Option<(Field, Field, Field)> {
    let field_const = |v: &Val| {
        ctx.const_vals
            .get(&v.0)
            .and_then(|cv| const_val_as_field(cv, ctx.field()))
    };
    Some((field_const(a)?, field_const(b)?, field_const(c)?))
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Val(ValueId);

struct SpecializationState<'a> {
    /// Shared reference to the SSA, used to mint fresh `ValueId`s via `fresh_value(&self)` while
    /// the symbolic executor holds its own shared `&HLSSA` borrow.
    ssa: &'a HLSSA,

    /// The candidate function body, mutated in place during symbolic execution. The candidate's
    /// `FunctionId` is owned by the caller. This is just the body slot, taken out of the SSA so it
    /// can be modified while the executor holds the SSA shared.
    body: HLFunction,

    /// Constant values created during specialization, usually by constant folding.
    const_vals: HashMap<ValueId, ConstVal>,

    /// The source location of the original instruction the symbolic executor is currently
    /// interpreting (via `Context::on_location`); residual instructions are located there.
    /// Starts at the function entry's first instruction location, covering anything emitted before
    /// execution reaches the first instruction.
    current_location: SourceLocation,
}

impl HLEmitter for SpecializationState<'_> {
    fn fresh_value(&mut self) -> ValueId {
        self.ssa.fresh_value()
    }

    fn emit(&mut self, instruction: OpCode) {
        let entry = self.body.get_entry_id();
        let location = self.current_location.clone();
        self.body
            .get_block_mut(entry)
            .push_instruction(instruction.locate(location));
    }

    fn emit_located(&mut self, instruction: LocatedOpCode) {
        let entry = self.body.get_entry_id();
        self.body.get_block_mut(entry).push_instruction(instruction);
    }

    fn emit_constant(&mut self, value: Constant) -> ValueId {
        self.ssa.add_const(value)
    }

    fn field(&self) -> FieldConfig {
        self.ssa.field()
    }
}

// FIELD-ASSUMPTION: L4-eval
impl symbolic_executor::Value<SpecializationState<'_>> for Val {
    /// `bits` is the caller's view of the operand width, and the fold reads its own off the
    /// `ConstVal::Int`s instead — the lattice needs a width from the constants anyway, and it folds
    /// only when the two operands agree on one. The two records must then _be_ the same width,
    /// which is what [`assert_recorded_width`] states: a signed comparison decodes at whichever it
    /// is told, and `Int(8, 0xFF)` is `-1` at 8 bits and `255` at 32. Asserted rather than
    /// preferred, because choosing between them would be choosing which of two disagreeing records
    /// to believe.
    fn cmp(
        &self,
        b: &Self,
        kind: CmpKind,
        bits: Option<usize>,
        ctx: &mut SpecializationState,
    ) -> Self {
        let l = const_val_scalar(ctx.const_vals.get(&self.0));
        let r = const_val_scalar(ctx.const_vals.get(&b.0));
        assert_recorded_width(bits, l.as_ref(), kind);
        assert_recorded_width(bits, r.as_ref(), kind);
        if let (Some(l), Some(r)) = (l, r)
            && let Some(folded) = lattice::eval_cmp(kind, &l, &r)
            && let Some(val) = intern_folded(ctx, folded)
        {
            return val;
        }
        // Re-emitted under the same `kind` it came in as.
        Self(ctx.cmp(self.0, b.0, kind))
    }

    fn arith(
        &self,
        b: &Self,
        binary_arith_op_kind: BinaryArithOpKind,
        _out_type: &Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let a_const = ctx.const_vals.get(&self.0).cloned();
        let b_const = ctx.const_vals.get(&b.0).cloned();

        // Every constant pair is folded by the shared lattice evaluator rather than by a second
        // implementation here. When it declines, control falls through to the match below, which
        // emits the operation unfolded.
        if let (Some(l), Some(r)) = (
            const_val_scalar(a_const.as_ref()),
            const_val_scalar(b_const.as_ref()),
        ) && folds_here(binary_arith_op_kind.group(), &l)
            && let Some(folded) = lattice::eval_binary(binary_arith_op_kind, &l, &r, ctx.field())
            && let Some(val) = intern_folded(ctx, folded)
        {
            return val;
        }

        // The identity rules below are this pass's own, not the lattice's: they fire on a pair
        // where only _one_ side is constant, which is not a fold at all.
        match (binary_arith_op_kind.group(), a_const, b_const) {
            (ArithGroup::Mul, Some(ConstVal::Field(f)), _) if f == ctx.field().one() => *b,
            (ArithGroup::Mul, _, Some(ConstVal::Field(f))) if f == ctx.field().one() => *self,
            (ArithGroup::Mul, Some(ConstVal::Field(f)), _) if f == ctx.field().zero() => *self,
            (ArithGroup::Mul, _, Some(ConstVal::Field(f))) if f == ctx.field().zero() => *b,

            (ArithGroup::Mul, None, None) => {
                let res = ctx.bin(binary_arith_op_kind, self.0, b.0);
                Self(res)
            }

            (ArithGroup::Add, Some(ConstVal::Field(f)), _) if f == ctx.field().zero() => *b,
            (ArithGroup::Add, _, Some(ConstVal::Field(f))) if f == ctx.field().zero() => *self,

            (ArithGroup::Add, _, _) => Self(ctx.bin(binary_arith_op_kind, self.0, b.0)),
            (ArithGroup::Sub, _, _) => Self(ctx.bin(binary_arith_op_kind, self.0, b.0)),
            (ArithGroup::Mul, _, _) => Self(ctx.bin(binary_arith_op_kind, self.0, b.0)),
            (ArithGroup::Div, _, _) => Self(ctx.bin(binary_arith_op_kind, self.0, b.0)),

            (ArithGroup::Rem, _, _) => Self(ctx.bin(binary_arith_op_kind, self.0, b.0)),

            (ArithGroup::And, _, _) => {
                let res = ctx.bin(binary_arith_op_kind, self.0, b.0);
                Self(res)
            }

            (ArithGroup::Or, _, _) => {
                let res = ctx.bin(binary_arith_op_kind, self.0, b.0);
                Self(res)
            }

            (ArithGroup::Xor, _, _) => {
                let res = ctx.bin(binary_arith_op_kind, self.0, b.0);
                Self(res)
            }

            // No constant-pair arm for any integer op: the lattice above folds every pair it
            // can, and one it declines must reach a catch-all and be emitted unfolded.
            (ArithGroup::Shl, _, _) => {
                let res = ctx.bin(binary_arith_op_kind, self.0, b.0);
                Self(res)
            }

            (ArithGroup::Shr, _, _) => {
                let res = ctx.bin(binary_arith_op_kind, self.0, b.0);
                Self(res)
            } /* No `_ => panic!("Not yet implemented")` arm: every op kind now ends in a catch-all
               * that emits the operation unfolded, so exhaustiveness checking proves an
               * unmodelled constant pair can no longer crash the compiler. Adding a variant to
               * `BinaryArithOpKind` will now fail to compile here rather than panicking at runtime. */
        }
    }

    fn assert_bool(&self, ctx: &mut SpecializationState) -> Result<(), AssertionFailure> {
        let v_const = ctx.const_vals.get(&self.0).cloned();
        // Truthiness is a property of the bit pattern, so this reads either tag. A non-integer
        // constant emits the assertion rather than panicking — declining to decide is always
        // safe, and the previous `panic!` made an unmodelled constant a compiler crash.
        match int_bits_and_raw(v_const.as_ref()) {
            Some((_, val)) => {
                if val == 0 {
                    return Err(AssertionFailure::new("assert failed: value is zero"));
                }
            }
            None => {
                HLEmitter::assert_bool(ctx, self.0);
            }
        }
        Ok(())
    }

    fn assert_cmp(
        kind: CmpKind,
        a: &Self,
        b: &Self,
        _bits: Option<usize>,
        ctx: &mut SpecializationState,
    ) -> Result<(), AssertionFailure> {
        let l_const = ctx.const_vals.get(&a.0);
        let r_const = ctx.const_vals.get(&b.0);
        match kind {
            CmpKind::Eq => match (l_const, r_const) {
                (Some(ConstVal::Int(_, l_val)), Some(ConstVal::Int(_, r_val))) => {
                    if l_val != r_val {
                        return Err(AssertionFailure::new(format!(
                            "assert_cmp eq failed: {l_val} != {r_val}"
                        )));
                    }
                }
                (None, _) | (_, None) => {
                    HLEmitter::assert_cmp(ctx, kind, a.0, b.0);
                }
                _ => panic!("Not yet implemented {:?}", (l_const, r_const)),
            },
            _ => {
                HLEmitter::assert_cmp(ctx, kind, a.0, b.0);
            }
        }
        Ok(())
    }

    fn assert_r1c(
        a: &Self,
        b: &Self,
        c: &Self,
        ctx: &mut SpecializationState,
    ) -> Result<(), AssertionFailure> {
        match r1c_consts(ctx, a, b, c) {
            Some((a, b, c)) => {
                if a * b != c {
                    return Err(AssertionFailure::new(format!(
                        "assert_r1c failed: {a:?} * {b:?} != {c:?}"
                    )));
                }
            }
            None => ctx.emit(OpCode::AssertR1C {
                a: a.0,
                b: b.0,
                c: c.0,
            }),
        }
        Ok(())
    }

    fn array_get(&self, index: &Self, _out_type: &Type, ctx: &mut SpecializationState) -> Self {
        let a_const = ctx.const_vals.get(&self.0).cloned();
        let index_const = ctx.const_vals.get(&index.0).cloned();
        match (a_const, index_const) {
            (Some(ConstVal::Array(a) | ConstVal::Blob(a)), Some(ConstVal::Int(_, index))) => {
                let res = a[index as usize];
                Self(res)
            }
            // FIELD-ASSUMPTION: L4-decompose
            (Some(ConstVal::BitsOf(v, size, endianness)), Some(ConstVal::Int(_, index))) => {
                let v_const = ctx.const_vals.get(v.as_ref()).cloned();
                match v_const {
                    Some(ConstVal::Field(f)) => {
                        let r = f.into_bigint().to_bits_le();
                        let ix = match endianness {
                            Endianness::Little => index as usize,
                            Endianness::Big => size - index as usize - 1,
                        };
                        let res = if r[ix] { 1 } else { 0 };
                        let res_v = ctx.int_const(1, res);
                        ctx.const_vals.insert(res_v, ConstVal::Int(1, res));
                        Self(res_v)
                    }
                    _ => panic!("Not yet implemented {:?}", (v_const, endianness)),
                }
            }
            (None, _) | (_, None) => {
                let res = HLEmitter::array_get(ctx, self.0, index.0);
                Self(res)
            }
            (a, i) => panic!("Not yet implemented {:?}", (a, i)),
        }
    }

    fn array_set(
        &self,
        _index: &Self,
        _value: &Self,
        _out_type: &Type,
        _ctx: &mut SpecializationState,
    ) -> Self {
        todo!()
    }

    fn sext(
        &self,
        from: usize,
        to: usize,
        _out_type: &Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        if let Some(c) = const_val_scalar(ctx.const_vals.get(&self.0))
            && let Some(folded) = lattice::eval_sext(&c, from, to)
            && let Some(val) = intern_folded(ctx, folded)
        {
            return val;
        }
        Self(ctx.sext(self.0, from, to))
    }

    fn bit_range(
        &self,
        offset: usize,
        width: usize,
        _out_type: &Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        if let Some(c) = const_val_scalar(ctx.const_vals.get(&self.0))
            && let Some(folded) = lattice::eval_bit_range(&c, offset, width, ctx.field())
            && let Some(val) = intern_folded(ctx, folded)
        {
            return val;
        }
        Self(ctx.bit_range(self.0, offset, width))
    }

    fn cast(
        &self,
        cast_target: &CastTarget,
        _out_type: &Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let Some(c) = const_val_scalar(ctx.const_vals.get(&self.0)) else {
            return Self(ctx.cast_to(cast_target.clone(), self.0));
        };
        let identity = match (cast_target, &c) {
            (
                CastTarget::Nop | CastTarget::ArrayToSlice | CastTarget::WitnessOf,
                Constant::Int(..) | Constant::Field(_),
            ) => true,
            (CastTarget::Field, Constant::Field(_)) => true,
            _ => false,
        };
        if identity {
            return *self;
        }

        if let Some(folded) = lattice::eval_cast(cast_target, &c, ctx.field())
            && let Some(val) = intern_folded(ctx, folded)
        {
            return val;
        }
        Self(ctx.cast_to(cast_target.clone(), self.0))
    }

    fn black_box(&self, _out_type: &Type, ctx: &mut SpecializationState) -> Self {
        // Emit a fresh barrier value and deliberately do not copy the input's lattice fact.
        Self(ctx.black_box(self.0))
    }

    fn constrain(
        a: &Self,
        b: &Self,
        c: &Self,
        ctx: &mut SpecializationState,
    ) -> Result<(), AssertionFailure> {
        match r1c_consts(ctx, a, b, c) {
            Some((a, b, c)) => {
                if a * b != c {
                    return Err(AssertionFailure::new(format!(
                        "constrain failed: {a:?} * {b:?} != {c:?}"
                    )));
                }
            }
            None => HLEmitter::constrain(ctx, a.0, b.0, c.0),
        }
        Ok(())
    }

    // FIELD-ASSUMPTION: L4-decompose
    fn to_bits(
        &self,
        endianness: Endianness,
        size: usize,
        _out_type: &Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let val = ctx.to_bits(self.0, endianness, size);
        ctx.const_vals
            .insert(val, ConstVal::BitsOf(Box::new(self.0), size, endianness));
        Self(val)
    }

    fn not(&self, _out_type: &Type, ctx: &mut SpecializationState) -> Self {
        if let Some(c) = const_val_scalar(ctx.const_vals.get(&self.0))
            && let Some(folded) = lattice::eval_not(&c)
            && let Some(val) = intern_folded(ctx, folded)
        {
            return val;
        }
        Self(HLEmitter::not(ctx, self.0))
    }

    fn of_int(s: usize, v: u128, ctx: &mut SpecializationState) -> Self {
        let val = ctx.int_const(s, v);
        ctx.const_vals.insert(val, ConstVal::Int(s, v));
        Self(val)
    }

    fn of_field(f: Field, ctx: &mut SpecializationState) -> Self {
        let val = ctx.field_const(f);
        ctx.const_vals.insert(val, ConstVal::Field(f));
        Self(val)
    }

    fn of_blob(elem_type: Type, elements: Vec<Self>, ctx: &mut SpecializationState) -> Self {
        fn constant_for(ctx: &SpecializationState<'_>, value: ValueId, typ: &Type) -> Constant {
            match ctx
                .const_vals
                .get(&value)
                .unwrap_or_else(|| panic!("Blob element v{} is not a constant", value.0))
            {
                ConstVal::Int(bits, value) => Constant::Int(*bits, *value),
                ConstVal::Field(value) => Constant::Field(*value),
                ConstVal::Blob(elements) => {
                    let inner = typ.get_array_element();
                    Constant::Blob(Blob::new(
                        inner.clone(),
                        elements
                            .iter()
                            .map(|element| constant_for(ctx, *element, &inner))
                            .collect(),
                    ))
                }
                other => panic!(
                    "Blob element v{} is not a scalar/blob constant: {:?}",
                    value.0, other
                ),
            }
        }

        let element_ids = elements.iter().map(|v| v.0).collect::<Vec<_>>();
        let constants = element_ids
            .iter()
            .map(|element| constant_for(ctx, *element, &elem_type))
            .collect();
        let val = ctx.emit_constant(Constant::Blob(Blob::new(elem_type, constants)));
        ctx.const_vals.insert(val, ConstVal::Blob(element_ids));
        Self(val)
    }

    fn expect_blob(&self, ctx: &mut SpecializationState) -> Vec<Self> {
        match ctx.const_vals.get(&self.0) {
            Some(ConstVal::Blob(elements)) => elements.iter().copied().map(Self).collect(),
            other => panic!("Expected blob, got {:?}", other),
        }
    }

    fn mk_array(
        a: Vec<Self>,
        ctx: &mut SpecializationState,
        seq_type: SequenceTargetType,
        elem_type: &Type,
    ) -> Self {
        let a = a.into_iter().map(|v| v.0).collect::<Vec<_>>();
        let val = ctx.mk_seq(a.clone(), seq_type, elem_type.clone());
        ctx.const_vals.insert(val, ConstVal::Array(a));
        Self(val)
    }

    fn alloc(value: &Self, ctx: &mut SpecializationState) -> Self {
        let val = ctx.alloc(value.0);
        Self(val)
    }

    fn ptr_write(&self, val: &Self, ctx: &mut SpecializationState) {
        ctx.store(self.0, val.0);
    }

    fn ptr_read(&self, _out_type: &Type, ctx: &mut SpecializationState) -> Self {
        let val = ctx.load(self.0);
        Self(val)
    }

    fn expect_constant_bool(&self, ctx: &mut SpecializationState) -> bool {
        let val = ctx.const_vals.get(&self.0).unwrap();
        match val {
            ConstVal::Int(_, v) => *v == 1,
            _ => todo!(),
        }
    }

    fn select(
        &self,
        if_t: &Self,
        if_f: &Self,
        _out_type: &Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let self_const = ctx.const_vals.get(&self.0);

        match self_const {
            Some(ConstVal::Int(_, v)) => {
                let res = if *v == 1 { if_t.0 } else { if_f.0 };
                Self(res)
            }
            None => {
                let res = HLEmitter::select(ctx, self.0, if_t.0, if_f.0);
                Self(res)
            }
            _ => todo!(),
        }
    }

    fn write_witness(&self, tp: Option<&Type>, ctx: &mut SpecializationState) -> Self {
        if ctx.const_vals.contains_key(&self.0) {
            return *self;
        }
        match tp {
            Some(_) => Self(HLEmitter::write_witness(ctx, self.0)),
            None => {
                ctx.emit(OpCode::WriteWitness {
                    result: None,
                    value: self.0,
                    pinned: false,
                });
                *self
            }
        }
    }

    fn fresh_witness(result_type: &Type, ctx: &mut SpecializationState) -> Self {
        let result = ctx.fresh_value();
        ctx.emit(OpCode::FreshWitness {
            result,
            result_type: result_type.clone(),
        });
        Self(result)
    }

    fn mem_op(&self, kind: RefCountOp, ctx: &mut SpecializationState) {
        HLEmitter::mem_op(ctx, self.0, kind);
    }

    fn rangecheck(
        &self,
        max_bits: usize,
        ctx: &mut SpecializationState,
    ) -> Result<(), AssertionFailure> {
        HLEmitter::rangecheck(ctx, self.0, max_bits);
        Ok(())
    }

    fn spread(&self, bits: u8, ctx: &mut SpecializationState) -> Self {
        let cst_val = ctx.const_vals.get(&self.0);
        match cst_val {
            Some(ConstVal::Int(b, v)) => {
                assert!(
                    *b <= 64,
                    "Spread only supports integer widths up to 64 bits, got int{}",
                    b
                );
                Self::of_int(b * 2, spread_bits(*v, *b), ctx)
            }
            _ => {
                let res = HLEmitter::spread(ctx, self.0, bits);
                Self(res)
            }
        }
    }

    fn unspread(&self, bits: u8, ctx: &mut SpecializationState) -> (Self, Self) {
        let cst_val = ctx.const_vals.get(&self.0);
        match cst_val {
            Some(ConstVal::Int(b, v)) => {
                assert!(
                    *b <= MAX_SUPPORTED_UNSIGNED_BITS && b % 2 == 0,
                    "Unspread expects an even integer width up to {MAX_SUPPORTED_UNSIGNED_BITS} bits, got int{}",
                    b
                );
                let half_bits = b / 2;
                let (odd, even) = unspread_bits(*v, *b);
                (
                    Self::of_int(half_bits, odd, ctx),
                    Self::of_int(half_bits, even, ctx),
                )
            }
            _ => {
                let (res_and, res_xor) = HLEmitter::unspread(ctx, self.0, bits);
                (Self(res_and), Self(res_xor))
            }
        }
    }

    fn to_radix(
        &self,
        radix: &Radix<Self>,
        endianness: Endianness,
        size: usize,
        _out_type: &Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let cst_val = ctx.const_vals.get(&self.0);
        match cst_val {
            None => {
                let radix = match radix {
                    Radix::Dyn(v) => Radix::Dyn(v.0),
                    Radix::Bytes => Radix::Bytes,
                };
                let res = HLEmitter::to_radix(ctx, self.0, radix, endianness, size);
                Self(res)
            }
            Some(_) => todo!(),
        }
    }
}

impl symbolic_executor::Context<Val> for SpecializationState<'_> {
    fn on_call(
        &mut self,
        func: FunctionId,
        params: &mut [Val],
        _param_types: &[&Type],
        result_types: &[Type],
        unconstrained: bool,
    ) -> Option<Vec<Val>> {
        if unconstrained {
            // Emit the unconstrained call as-is into the specialized function
            let args: Vec<ValueId> = params.iter().map(|v| v.0).collect();
            let n = result_types.len();
            let results = self.call_unconstrained(func, args, n);
            return Some(results.into_iter().map(Val).collect());
        }
        None
    }

    fn on_return(&mut self, returns: &mut [Val], _return_types: &[Type]) {
        self.body.terminate_block_with_return(
            self.body.get_entry_id(),
            returns.iter().map(|v| v.0).collect(),
        );
    }

    fn on_jmp(&mut self, _target: BlockId, _params: &mut [Val], _param_types: &[&Type]) {}

    fn on_location(&mut self, location: &SourceLocation) {
        self.current_location = location.clone();
    }

    fn lookup(&mut self, target: LookupTarget<Val>, args: Vec<Val>, flag: Val) {
        self.emit(OpCode::Lookup {
            target: target.map(|v| v.0),
            args: args.into_iter().map(|arg| arg.0).collect(),
            flag: flag.0,
        });
    }

    fn dlookup(&mut self, target: LookupTarget<Val>, args: Vec<Val>, flag: Val) {
        self.emit(OpCode::DLookup {
            target: target.map(|v| v.0),
            args: args.into_iter().map(|arg| arg.0).collect(),
            flag: flag.0,
        });
    }

    fn todo(&mut self, payload: &str, _result_types: &[Type]) -> Vec<Val> {
        todo!("Todo opcode: {}", payload);
    }

    fn slice_len(&mut self, slice: &Val) -> Val {
        if let Some(ConstVal::Array(elements)) = self.const_vals.get(&slice.0) {
            let len = elements.len() as u128;
            let val = self.int_const(32, len);
            self.const_vals.insert(val, ConstVal::Int(32, len));
            Val(val)
        } else {
            let val = HLEmitter::slice_len(self, slice.0);
            Val(val)
        }
    }

    fn on_guard(
        &mut self,
        inner: &OpCode,
        condition: &Val,
        inputs: Vec<&Val>,
        _result_types: Vec<&Type>,
    ) -> Vec<Val> {
        use crate::compiler::ssa::Instruction;

        // Build a mapping from old ValueIds to new ValueIds
        let orig_inputs: Vec<_> = inner.get_inputs().cloned().collect();
        let orig_results: Vec<_> = inner.get_results().cloned().collect();
        let mut id_map: HashMap<ValueId, ValueId> = HashMap::default();
        for (orig, new_val) in orig_inputs.iter().zip(inputs.iter()) {
            id_map.insert(*orig, new_val.0);
        }
        let mut result_vals = Vec::new();
        for orig_result in &orig_results {
            let fresh = self.fresh_value();
            id_map.insert(*orig_result, fresh);
            result_vals.push(Val(fresh));
        }
        // Clone and remap all operands
        let mut new_inner = inner.clone();
        for op in new_inner.get_operands_mut() {
            if let Some(new_id) = id_map.get(op) {
                *op = *new_id;
            }
        }
        self.emit(OpCode::Guard {
            condition: condition.0,
            inner: Box::new(new_inner),
        });
        result_vals
    }
}

impl Pass for Specializer {
    fn name(&self) -> &'static str {
        "specializer"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![Summary::id(), TypeInfo::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        let summary = store.get::<Summary>();
        let mut speculative_ids: HashSet<FunctionId> = HashSet::default();
        let mut accepted_ids: HashSet<FunctionId> = HashSet::default();

        for (sig, summary) in summary.functions.iter() {
            if summary.specialization_total_savings > 0 {
                self.try_spec(
                    ssa,
                    store.get::<TypeInfo>(),
                    summary,
                    sig.clone(),
                    &mut speculative_ids,
                    &mut accepted_ids,
                );
            }
        }

        // Drop any speculative candidate (and its `#unspecialized` clone, if it had one)
        // that wasn't accepted. Constants the rejected candidates left behind become
        // unreferenced and are cleaned up by the DCE pass that runs immediately after the
        // specializer.
        ssa.retain_functions(|id, _| !speculative_ids.contains(&id) || accepted_ids.contains(&id));
    }
}

impl Specializer {
    pub fn new(savings_to_code_ratio: f64) -> Self {
        Self {
            savings_to_code_ratio,
        }
    }

    #[instrument(skip_all, name = "Specializer::try_spec", fields(function = %signature.pretty_print(ssa, true), expected_savings = summary.specialization_total_savings))]
    fn try_spec(
        &self,
        ssa: &mut HLSSA,
        type_info: &TypeInfo,
        summary: &SpecializationSummary,
        signature: FunctionSignature,
        speculative_ids: &mut HashSet<FunctionId>,
        accepted_ids: &mut HashSet<FunctionId>,
    ) {
        let name = signature.pretty_print(ssa, true);

        if summary.specialization_total_savings as f64 / self.savings_to_code_ratio < 10.0 {
            info!(
                message = %"Specialization rejected, would need less than 10 codesize to be worth it",
                specialization = %name,
                saved_constraints = summary.specialization_total_savings,
                savings_to_code_ratio = self.savings_to_code_ratio
            );
            return;
        }

        // Snapshot what we need from the original before any mutation: param types, return
        // types, and its name (for the #specialized / #unspecialized derived names).
        let original_param_types: Vec<Type>;
        let original_return_types: Vec<Type>;
        let original_name: String;
        {
            let original_fn = ssa.get_function(signature.get_fun_id());
            original_param_types = original_fn.get_param_types();
            original_return_types = original_fn.get_returns().to_vec();
            original_name = original_fn.get_name().to_string();
        }

        // Mint the candidate's FunctionId up-front. The empty body lives in the SSA at this
        // id until we put the filled-in one back. Track it as speculative so the end-of-pass
        // cleanup drops it if the specialization is ultimately rejected.
        let candidate_id = ssa.add_function(name.clone());
        speculative_ids.insert(candidate_id);

        // Take the empty body out so the state can mutate it while the symbolic executor
        // holds a shared `&HLSSA`.
        let mut body = ssa.take_function(candidate_id);
        for ret in &original_return_types {
            body.add_return_type(ret.clone());
        }

        // Build call params and the initial `const_vals` map. Routes through `add_const`
        // (still `&mut ssa` here) so the constants for `Field`/`U`/`I` signature params are
        // interned eagerly.
        let mut call_params: Vec<Val> = vec![];
        let mut const_vals: HashMap<ValueId, ConstVal> = HashMap::default();
        for (param, sig) in original_param_types
            .iter()
            .zip(signature.get_params().iter())
        {
            match sig {
                ValueSignature::PointerTo(_) => {
                    info!("TODO: Aborting specialization on a pointer value");
                    return;
                }
                ValueSignature::Array(_) => {
                    info!("TODO: Aborting specialization on an array value");
                    return;
                }
                ValueSignature::Blob(_) => {
                    info!("TODO: Aborting specialization on a blob value");
                    return;
                }
                ValueSignature::Unknown(_)
                | ValueSignature::UnknownSlice
                | ValueSignature::WitnessOf(_) => {
                    let id = ssa.fresh_value();
                    body.get_entry_mut().push_parameter(id, param.clone());
                    call_params.push(Val(id));
                }
                ValueSignature::Field(f) => {
                    let val = ssa.add_const(Constant::Field(*f));
                    call_params.push(Val(val));
                    const_vals.insert(val, ConstVal::Field(*f));
                }
                ValueSignature::Int { bits_size, value } => {
                    let val = ssa.add_const(Constant::Int(*bits_size, *value));
                    call_params.push(Val(val));
                    const_vals.insert(val, ConstVal::Int(*bits_size, *value));
                }
            }
        }

        let body = {
            let current_location = body
                .get_entry()
                .first_location()
                .cloned()
                .unwrap_or_else(|| SourceLocation::synthetic("specializer"));
            let mut state = SpecializationState {
                ssa: &*ssa,
                body,
                const_vals,
                current_location,
            };

            // Specialization is speculative: this candidate may sit behind a branch that never
            // executes at runtime. A statically-violated assertion encountered while folding
            // these constant arguments therefore does NOT mean the whole program fails — it
            // only means this particular specialization is invalid. Abort the candidate (it is
            // cleaned up by the end-of-pass `retain_functions`) and leave the unspecialized
            // function in place; if the assertion is genuinely unreachable-free, R1CS
            // generation will rediscover and report it.
            if let Err(failure) = SymbolicExecutor::new().run(
                &*ssa,
                type_info,
                signature.get_fun_id(),
                call_params,
                &mut state,
            ) {
                info!(
                    message = %"Aborting specialization: assertion is statically violated for these arguments",
                    specialization = %name,
                    failure = %failure
                );
                return;
            }

            state.body
        };

        let code_bloat = body.code_size();
        let savings_to_code_ratio = summary.specialization_total_savings as f64 / code_bloat as f64;

        // Put the body back unconditionally. On rejection it stays at `candidate_id` only
        // until the end-of-pass `retain_functions` call drops it.
        ssa.put_function(candidate_id, body);

        if savings_to_code_ratio > self.savings_to_code_ratio {
            info!(message = %"Specialization accepted", code_bloat = code_bloat,  savings_to_code_ratio = savings_to_code_ratio, threshold_ratio = self.savings_to_code_ratio);

            // Clone the original via the SSA helper. The clone has fresh `ValueId`s and
            // becomes the dispatcher's fallback target; the original's slot is then
            // overwritten with the dispatcher itself.
            let unspecialized_id = ssa.duplicate_function(signature.get_fun_id());
            ssa.get_function_mut(unspecialized_id)
                .set_name(format!("{}#unspecialized", original_name));

            let dispatcher = self.build_dispatcher_for(
                ssa,
                original_param_types,
                original_return_types,
                &signature,
                format!("{}#specialized", original_name),
                candidate_id,
                unspecialized_id,
            );
            *ssa.get_function_mut(signature.get_fun_id()) = dispatcher;

            accepted_ids.insert(candidate_id);
            accepted_ids.insert(unspecialized_id);
        } else {
            info!(message = %"Specialization rejected", code_bloat = code_bloat,  savings_to_code_ratio = savings_to_code_ratio, threshold_ratio = self.savings_to_code_ratio);
        }
    }

    fn build_dispatcher_for(
        &self,
        ssa: &mut HLSSA,
        params: Vec<Type>,
        returns: Vec<Type>,
        signature: &FunctionSignature,
        fn_name: String,
        specialized_id: FunctionId,
        unspecialized_id: FunctionId,
    ) -> HLFunction {
        let location = SourceLocation::synthetic(&fn_name);
        let mut dispatcher = HLFunction::empty(fn_name);
        let entry_block = dispatcher.get_entry_id();

        let mut b = HLFunctionBuilder::new(&mut dispatcher, ssa);

        let mut dispatcher_params = vec![];
        {
            let mut entry = b.block(entry_block);
            for param in params {
                dispatcher_params.push(entry.add_parameter(param));
            }
        }

        for return_type in returns.iter() {
            b.function().add_return_type(return_type.clone());
        }

        let mut specialized_params = vec![];
        let should_call_spec;
        {
            let mut entry = b.block(entry_block).with_source_location(location.clone());
            let mut cond = entry.int_const(1, 1);

            for (pval, psig) in dispatcher_params.iter().zip(signature.get_params().iter()) {
                match psig {
                    ValueSignature::PointerTo(_) => {
                        unreachable!(
                            "ICE: pointer specializations are rejected before dispatcher generation"
                        );
                    }
                    ValueSignature::Array(_) => {
                        unreachable!(
                            "ICE: array specializations are rejected before dispatcher generation"
                        );
                    }
                    ValueSignature::Blob(_) => {
                        unreachable!(
                            "ICE: blob specializations are rejected before dispatcher generation"
                        );
                    }
                    ValueSignature::Unknown(_)
                    | ValueSignature::UnknownSlice
                    | ValueSignature::WitnessOf(_) => {
                        specialized_params.push(*pval);
                    }
                    ValueSignature::Field(v) => {
                        let cst = entry.field_const(*v);
                        let is_eq = entry.eq(*pval, cst);
                        cond = entry.and(cond, is_eq);
                    }
                    ValueSignature::Int { bits_size, value } => {
                        let cst = entry.int_const(*bits_size, *value);
                        let is_eq = entry.eq(*pval, cst);
                        cond = entry.and(cond, is_eq);
                    }
                }
            }
            should_call_spec = cond;
        }

        let specialized_caller = b.add_block(|_| {});
        let unspecialized_caller = b.add_block(|_| {});

        let mut return_values = vec![];
        let return_block = b.add_block(|ret| {
            for r in returns {
                return_values.push(ret.add_parameter(r));
            }
            ret.terminate_return(return_values.clone());
        });

        {
            let mut cb = b
                .block(unspecialized_caller)
                .with_source_location(location.clone());
            let unspecialized_returns =
                cb.call(unspecialized_id, dispatcher_params, return_values.len());
            cb.terminate_jmp(return_block, unspecialized_returns);
        }

        {
            let mut cb = b.block(specialized_caller).with_source_location(location);
            let specialized_returns =
                cb.call(specialized_id, specialized_params, return_values.len());
            cb.terminate_jmp(return_block, specialized_returns);
        }

        b.block(entry_block).terminate_jmp_if(
            should_call_spec,
            specialized_caller,
            unspecialized_caller,
        );

        dispatcher
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{analysis::symbolic_executor::Value as _, located::SourceLocation};

    /// Fold `a << b` exactly as the specializer does, returning the constant it produced or `None`
    /// when it declined and left the shift in the residual program.
    ///
    /// This arm is **not reachable on the corpus** — probed across nine shift- and
    /// specialization-heavy tests, none of which reaches it — so a unit test is the only thing
    /// that can hold it to the contract.
    fn specializer_shl(a: (usize, u128), b: (usize, u128)) -> Option<(usize, u128)> {
        specializer_fold(BinaryArithOpKind::UShl, a, b)
    }

    /// As [`specializer_shl`], for any binary arithmetic opcode.
    fn specializer_fold(
        kind: BinaryArithOpKind,
        a: (usize, u128),
        b: (usize, u128),
    ) -> Option<(usize, u128)> {
        let mut ssa = HLSSA::new();
        let fid = ssa
            .get_function_ids()
            .next()
            .expect("HLSSA::new makes a main");
        let body = ssa.take_function(fid);

        let lhs = ssa.add_const(Constant::Int(a.0, a.1));
        let rhs = ssa.add_const(Constant::Int(b.0, b.1));
        let mut ctx = SpecializationState {
            ssa: &ssa,
            body,
            const_vals: HashMap::default(),
            current_location: SourceLocation::synthetic("specializer_test"),
        };
        ctx.const_vals.insert(lhs, ConstVal::Int(a.0, a.1));
        ctx.const_vals.insert(rhs, ConstVal::Int(b.0, b.1));

        let out = Val(lhs).arith(&Val(rhs), kind, &Type::int(a.0), &mut ctx);
        match ctx.const_vals.get(&out.0) {
            Some(ConstVal::Int(s, v)) => Some((*s, *v)),
            // Declined: `arith` emitted the shift instead, and its result has no constant.
            _ => None,
        }
    }

    #[test]
    fn unsigned_shl_folds_to_a_constant_inside_its_own_width() {
        // The bug this pins: `1 << 32` at `u32` used to fold to `Constant::Int(32, 4294967296)`, a
        // value one bit wider than the type it is tagged with. `hlssa_to_r1cs` wraps such a
        // constant on the way in and reads `0` while the VM's `mov_const` carries the whole
        // payload, so the two backends disagree about the program.
        //
        // The amount is what fails, not the value, so `1 << 32` at `u32` does not fold at all.
        assert_eq!(specializer_shl((32, 1), (32, 32)), None);
        assert_eq!(specializer_shl((8, 1), (8, 8)), None);

        // A shift that stays inside the width is unaffected.
        assert_eq!(specializer_shl((32, 1), (32, 31)), Some((32, 1 << 31)));
        assert_eq!(specializer_shl((64, 1), (64, 32)), Some((64, 1 << 32)));
        assert_eq!(specializer_shl((8, 1), (8, 0)), Some((8, 1)));

        // And one that leaves it **wraps** rather than being refused or widened -- `200 << 1` is
        // `400`, which keeps only its low eight bits. Every value here is the one
        // `noir_tests/pure_guarded_shift` checks against Noir's own execution.
        assert_eq!(specializer_shl((8, 200), (8, 1)), Some((8, 144)));
        assert_eq!(specializer_shl((8, 200), (8, 7)), Some((8, 0)));
        assert_eq!(specializer_shl((8, 255), (8, 7)), Some((8, 128)));
    }

    #[test]
    fn unsigned_shl_declines_a_mixed_width_pair_in_either_direction() {
        // Neither direction folds, and the reason is the same both ways: `assert_int_arith_widths`
        // requires an integer `BinaryArithOp`'s operands to be exactly the width of its result, so
        // a mixed-width shift is IR nothing may build and there is no constant to mint for it. A
        // wider amount would additionally narrow the result -- the type analysis types it as
        // `U(max(s1, s2))` -- but that is a second reason rather than the deciding one.
        //
        // The narrower-amount direction used to fold, on the grounds that the *model* gives a shift
        // amount a width of its own. It does, and that freedom is real for the evaluators that meet
        // one at runtime; it is not a licence for a folder to mint IR the rest of the pipeline
        // rejects.
        assert_eq!(specializer_shl((8, 1), (32, 1)), None);
        assert_eq!(specializer_shl((32, 1), (8, 1)), None);
    }

    #[test]
    fn an_integer_fold_that_leaves_its_width_is_refused_rather_than_widened_or_panicking() {
        use BinaryArithOpKind::{SDiv, UAdd, UDiv, UShr, USub};

        // Until the constant tags collapsed, these arms were seven hand-rolled `u128` operations
        // reachable only for a pair that happened to be tagged `U`. They are the lattice's job now,
        // which is what fixes all three of the following. Each line fails with the old arms
        // restored -- the first by producing an over-wide constant, the second and third by
        // _panicking_ in a debug build rather than declining.

        // `u8 200 + 100` is 300, which does not fit in eight bits.
        assert_eq!(specializer_fold(UAdd, (8, 200), (8, 100)), None);
        // `u8 5 - 10` underflows. `a_val - b_val` on raw `u128`s panics on the spot.
        assert_eq!(specializer_fold(USub, (8, 5), (8, 10)), None);
        // A shift amount at or past the width is a runtime error in Noir, so it must not fold to
        // anything; `a_val >> b_val` with `b_val >= 128` panics.
        assert_eq!(specializer_fold(UShr, (8, 0xFF), (8, 200)), None);

        // Folds that stay in range are unaffected...
        assert_eq!(specializer_fold(UAdd, (8, 40), (8, 2)), Some((8, 42)));
        assert_eq!(specializer_fold(USub, (8, 40), (8, 2)), Some((8, 38)));
        assert_eq!(specializer_fold(UShr, (8, 0xFF), (8, 4)), Some((8, 0x0F)));

        // ...and the opcode is what picks the reading, on operands that carry none. 0xFB is -5 as
        // two's complement and 251 as a magnitude: -5 / 2 is -2 (0xFE), 251 / 2 is 125 (0x7D).
        assert_eq!(specializer_fold(SDiv, (8, 0xFB), (8, 2)), Some((8, 0xFE)));
        assert_eq!(specializer_fold(UDiv, (8, 0xFB), (8, 2)), Some((8, 0x7D)));
    }
}
