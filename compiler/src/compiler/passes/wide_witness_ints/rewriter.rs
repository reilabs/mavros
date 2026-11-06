use mavros_artifacts::FieldConfig;
use mavros_int_semantics::IntBits;

use super::globals::Globals;
use super::types::{flat_limbs, limb_types, limb_widths, limbs_of, single, wide_witness_width};
use crate::collections::HashMap;
use crate::compiler::{
    Field,
    analysis::{
        types::{FunctionTypeInfo, TypeInfo},
        value_range_analysis::field_modulus,
    },
    passes::shared::limbs::{widest_injective_int_bits, witness_limb_bits},
    ssa::{
        BlockId, FunctionId, Located, Terminator, ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, Constant, HLSSA, LocatedOpCode, OpCode,
            builder::HLEmitter,
        },
    },
    util::field_constant,
};
use chain::{CarryOut, chained};
use division::{Answer, Division, divided};
use product::multiplied;
use shift::shifted;
mod cast;
mod chain;
mod division;
mod gadgets;
mod lower;
mod product;
mod shift;
mod sign;
mod spread;
#[cfg(test)]
pub(super) use product::division_carries;
/// One instruction's worth of rewriting, plus the minting the gadgets need.
///
/// `ssa` is borrowed immutably, which is enough to mint a value or intern a constant: the function
/// being rewritten is out of the SSA while it is rewritten.
struct Rewriter<'a> {
    ssa: &'a HLSSA,
    value_map: &'a HashMap<ValueId, Vec<ValueId>>,
    types: &'a FunctionTypeInfo,

    field: FieldConfig,

    /// The decompositions already emitted in this block, by value and width.
    decomposed: &'a mut HashMap<(ValueId, usize), Vec<ValueId>>,

    /// The top bits already cut out of a witnessed limb in this block, by limb and width.
    signs: &'a mut HashMap<(ValueId, usize), ValueId>,

    /// The divisions already built in this block, as their quotient and remainder, by dividend,
    /// divisor, guard and reading.
    divisions: &'a mut HashMap<Division, (Answer, Answer)>,

    /// The limbs a pure integer read in this block was recombined from, by the read's result.
    ///
    /// A user that wants the limbs back, as the field image a witness-indexed lookup moves through
    /// does, takes these rather than splitting what was just joined. The split uses a logical
    /// right shift (except at offset zero) and a narrowing per limb at the value's full width,
    /// undoing the recombination the read just performed.
    read_limbs: &'a mut HashMap<ValueId, Vec<ValueId>>,

    /// The values this function's rewrite has minted whose pattern is known at compile time but
    /// that are not constants themselves, by value.
    known: &'a mut HashMap<ValueId, IntBits>,

    /// Where each global's slots moved to.
    globals: &'a Globals,

    out: Vec<OpCode>,
}

pub(super) fn rewrite_function(
    ssa: &mut HLSSA,
    fid: FunctionId,
    reachable: &[BlockId],
    value_map: &HashMap<ValueId, Vec<ValueId>>,
    type_info: &TypeInfo,
    field: FieldConfig,
    globals: &Globals,
) {
    let mut function = ssa.take_function(fid);
    let fti = type_info.get_function(fid);

    let old_returns = function.take_returns();
    for ty in old_returns {
        for limb in limb_types(&ty, field) {
            function.add_return_type(limb);
        }
    }

    let mut known = HashMap::default();
    for bid in reachable {
        let block = function.get_block_mut(*bid);

        let old_params = block.take_parameters();
        let mut new_params = Vec::new();
        for (pid, ty) in old_params {
            let ids = limbs_of(value_map, pid);
            let types = limb_types(&ty, field);
            debug_assert_eq!(ids.len(), types.len());
            for (id, ty) in ids.into_iter().zip(types) {
                new_params.push((id, ty));
            }
        }
        block.put_parameters(new_params);

        let old_instructions = block.take_instructions();
        let mut new_instructions = Vec::with_capacity(old_instructions.len());
        let mut decomposed = HashMap::default();
        let mut signs = HashMap::default();
        let mut divisions = HashMap::default();
        let mut read_limbs = HashMap::default();
        for instr in &old_instructions {
            let location = instr.location().clone();
            let mut rewriter = Rewriter {
                ssa,
                value_map,
                types: fti,
                field,
                decomposed: &mut decomposed,
                signs: &mut signs,
                divisions: &mut divisions,
                read_limbs: &mut read_limbs,
                known: &mut known,
                globals,
                out: Vec::new(),
            };
            rewriter.lower(instr.as_ref());
            new_instructions.extend(
                rewriter
                    .out
                    .into_iter()
                    .map(|op| Located::new(op, location.clone())),
            );
        }
        block.put_instructions(new_instructions);

        let terminator = match block.take_terminator().unwrap() {
            Terminator::Jmp(dest, args) => Terminator::Jmp(dest, flat_limbs(value_map, &args)),
            Terminator::JmpIf(cond, t, f) => Terminator::JmpIf(single(value_map, cond), t, f),
            Terminator::Return(values) => Terminator::Return(flat_limbs(value_map, &values)),
        };
        block.set_terminator(terminator);
    }

    ssa.put_function(fid, function);
}

impl Rewriter<'_> {
    fn limb_bits(&self) -> usize {
        witness_limb_bits(self.field)
    }

    fn limbs(&self, value: ValueId) -> Vec<ValueId> {
        limbs_of(self.value_map, value)
    }

    /// The limbs of an operand, splitting a **pure** one the plan never saw.
    ///
    /// `check_widths` makes both operands of a non-shift the same width, and a mixed pure/witness
    /// pair is legal, so one witnessed operand is sufficient to require a witness lowering. A
    /// constant is split at compile time, so its limbs are constants too. A pure value that a
    /// sequence read in this block recombined hands back the limbs it was recombined from.
    fn operand_limbs(&mut self, value: ValueId, expected: usize) -> Vec<ValueId> {
        let mapped = self.limbs(value);
        if mapped.len() == expected {
            return mapped;
        }
        if let Some(limbs) = self.read_limbs.get(&value)
            && limbs.len() == expected
        {
            return limbs.clone();
        }
        assert_eq!(
            mapped.len(),
            1,
            "ICE: an operand of {} limbs met one of {expected}",
            mapped.len()
        );

        let bits = self
            .types
            .get_value_type(value)
            .int_width()
            .unwrap_or_else(|| ice!("a non-integer operand met a wide witnessed integer"));
        let widths = limb_widths(bits, self.limb_bits());
        assert_eq!(
            widths.len(),
            expected,
            "ICE: an int{bits} operand met a value of {expected} limbs"
        );

        let constant = self.known(value);
        widths
            .into_iter()
            .enumerate()
            .map(|(index, width)| match &constant {
                Some(pattern) => self.int_const(pattern.bit_range(index * self.limb_bits(), width)),
                None => {
                    let shifted = self.shifted_down(value, bits, index * self.limb_bits());
                    self.cast(shifted, CastTarget::Int(width))
                }
            })
            .collect()
    }

    /// The limbs of one operation's two operands, which are the same length by construction.
    fn operand_pair(&mut self, lhs: ValueId, rhs: ValueId) -> Vec<(ValueId, ValueId)> {
        let expected = self.limbs(lhs).len().max(self.limbs(rhs).len());
        let lhs = self.operand_limbs(lhs, expected);
        let rhs = self.operand_limbs(rhs, expected);
        lhs.into_iter().zip(rhs).collect()
    }

    fn one(&self, value: ValueId) -> ValueId {
        single(self.value_map, value)
    }

    /// The guard's condition, as its one limb, and the operation it wraps, or no condition and `op`
    /// itself where there is no guard.
    fn split_guard<'o>(&self, op: &'o OpCode) -> (Option<ValueId>, &'o OpCode) {
        match op {
            OpCode::Guard { condition, inner } => (Some(self.one(*condition)), inner.as_ref()),
            other => (None, other),
        }
    }

    /// The declared width of a witnessed integer this pass represents as limbs.
    fn wide_width(&self, value: ValueId) -> Option<usize> {
        wide_witness_width(self.types.get_value_type(value), self.field)
    }

    fn fresh(&self) -> ValueId {
        self.ssa.fresh_value()
    }

    fn push(&mut self, op: OpCode) {
        if let OpCode::Cast {
            result,
            value,
            target,
        } = &op
            && let Some(pattern) = self.known(*value)
        {
            let pattern = match target {
                CastTarget::WitnessOf | CastTarget::ValueOf | CastTarget::Nop => Some(pattern),
                CastTarget::Int(bits) => Some(pattern.cast(*bits)),
                _ => None,
            };
            if let Some(pattern) = pattern {
                self.known.insert(*result, pattern);
            }
        }
        self.out.push(op);
    }

    /// The pattern of an integer known at compile time: a constant, or a value minted from one
    /// through casts that keep or cut its pattern.
    fn known(&self, value: ValueId) -> Option<IntBits> {
        match self.ssa.get_const(value).as_deref() {
            Some(Constant::Int(pattern)) => Some(pattern.clone()),
            Some(_) => None,
            None => self.known.get(&value).cloned(),
        }
    }

    fn cast(&mut self, value: ValueId, target: CastTarget) -> ValueId {
        let result = self.fresh();
        self.push(OpCode::Cast {
            result,
            value,
            target,
        });
        result
    }

    /// `value >> offset` on the pure side.
    ///
    /// Up to the injective width it is a division by `2^offset`: `Simplifier` turns a shift by a
    /// constant there into a `BitRange`, and it runs after the rule that lowers `BitRange`, so the
    /// window would reach codegen and be refused.
    ///
    /// Past that width it is a shift, which the simplifier leaves alone. Wasm emits it inline,
    /// where a division calls a runtime helper per limb and, at 16384 bits, makes the function too
    /// large to compile.
    fn shifted_down(&mut self, value: ValueId, bits: usize, offset: usize) -> ValueId {
        if offset == 0 {
            return value;
        }
        if bits > widest_injective_int_bits(self.field) {
            let amount = self.int_const(IntBits::from_u128(bits, offset as u128));
            return self.bin(BinaryArithOpKind::UShr, value, amount);
        }
        let divisor = self.two_pow_const(bits, offset);
        self.bin(BinaryArithOpKind::UDiv, value, divisor)
    }

    /// `op`, under `guard` where there is one.
    fn push_guarded(&mut self, guard: Option<ValueId>, op: OpCode) {
        match guard {
            Some(condition) => self.push(OpCode::Guard {
                condition,
                inner: Box::new(op),
            }),
            None => self.push(op),
        }
    }

    fn is_witness(&self, value: ValueId) -> bool {
        self.types.get_value_type(value).is_witness_of()
    }

    fn field_const(&self, value: Field) -> ValueId {
        self.ssa.add_const(Constant::Field(value))
    }

    /// A limb's pattern as a field constant.
    fn limb_const(&self, pattern: &IntBits) -> ValueId {
        self.field_const(
            field_constant(self.field, pattern)
                .unwrap_or_else(|| ice!("a limb's pattern {pattern:?} is wider than the field")),
        )
    }

    /// `2^(index * h)` as a field element, the place value of limb `index`.
    fn place_value(&self, index: usize) -> ValueId {
        self.field_const(self.field.two_pow(index * self.limb_bits()))
    }

    /// A witnessed field element read as an integer of `to_bits` held as limbs, into `results`.
    ///
    /// An element below the modulus fits any width the representation holds, so this is its
    /// canonical value, zero-filled. It is decomposed at the field's width by
    /// [`Self::witness_limbs_of`]: limbs from the pure shadow, each range-checked at its width,
    /// constrained to sum to the element. That sum is taken modulo `p` and the limbs span past it,
    /// so `v` and `v + p` both satisfy it where `v + p` fits the limbs. The carry chain with its
    /// top borrow forced out holds the limbs below the modulus, leaving only the canonical value.
    fn lower_field_into_limbs(&mut self, value: ValueId, to_bits: usize, results: &[ValueId]) {
        let field_bits = self.field.field_bit_size() as usize;
        let widths = limb_widths(field_bits, self.limb_bits());
        let shadow = self.cast(value, CastTarget::ValueOf);
        let pure = self.cast(shadow, CastTarget::Int(field_bits));
        let (limbs, sum) = self.witness_limbs_of(pure, field_bits);
        self.constrain_equal(value, sum, None);

        let modulus = IntBits::from_biguint(
            field_bits,
            &field_modulus(self.field)
                .to_biguint()
                .expect("a modulus is positive"),
        );
        let witnessed: Vec<(ValueId, bool)> = limbs.iter().map(|limb| (*limb, true)).collect();
        let bound: Vec<(ValueId, bool)> = widths
            .iter()
            .enumerate()
            .map(|(index, width)| {
                let limb = modulus.bit_range(index * self.limb_bits(), *width);
                (self.int_const(limb), false)
            })
            .collect();
        self.carry_chain_over(true, &witnessed, &bound, widths, CarryOut::One, None);

        let target_widths = limb_widths(to_bits, self.limb_bits());
        assert_eq!(
            target_widths.len(),
            results.len(),
            "ICE: an int{to_bits} is {} limbs, not {}",
            target_widths.len(),
            results.len()
        );
        for (index, (result, width)) in results.iter().zip(target_widths).enumerate() {
            match limbs.get(index) {
                Some(limb) => self.push(OpCode::Cast {
                    result: *result,
                    value: *limb,
                    target: CastTarget::Int(width),
                }),
                None => self.define_zero(*result, width),
            }
        }
    }

    /// Limb `index` of a pure `bits`-wide `value`, at `width`, written to a witness column and
    /// range-checked at `checked` bits. Returns the limb and its field element.
    ///
    /// Written from the hint, not injected: an injection keeps the pure shadow live into R1CS,
    /// where a witness strip is an ICE.
    fn witness_limb(
        &mut self,
        value: ValueId,
        bits: usize,
        index: usize,
        width: usize,
        checked: usize,
    ) -> (ValueId, ValueId) {
        let hint = self.shifted_down(value, bits, index * self.limb_bits());
        let narrowed = self.cast(hint, CastTarget::Int(width));
        let hint_field = self.cast(narrowed, CastTarget::Field);
        let written = self.write_witness(hint_field);
        let limb = self.cast(written, CastTarget::Int(width));
        let limb_field = self.cast(limb, CastTarget::Field);
        self.push(OpCode::Rangecheck {
            value: limb_field,
            max_bits: checked,
        });
        (limb, limb_field)
    }

    /// `sum + element · 2^(h·index)` in one field element.
    fn add_at_place(&mut self, sum: Option<ValueId>, element: ValueId, index: usize) -> ValueId {
        let scaled = if index == 0 {
            element
        } else {
            let place = self.place_value(index);
            self.bin(BinaryArithOpKind::UMul, element, place)
        };
        match sum {
            None => scaled,
            Some(acc) => self.bin(BinaryArithOpKind::UAdd, acc, scaled),
        }
    }

    /// `Σ limbᵢ · 2^(h·i)` in one field element: the integer the limbs spell, reduced modulo the
    /// field's modulus.
    fn limb_sum(&mut self, limbs: &[ValueId]) -> ValueId {
        let mut sum = None;
        for (index, limb) in limbs.iter().enumerate() {
            let limb_field = self.cast(*limb, CastTarget::Field);
            sum = Some(self.add_at_place(sum, limb_field, index));
        }
        sum.expect("an integer has at least one limb")
    }

    /// The low `width` bits of a limb, as a value **of that width**.
    ///
    /// A narrowing cast, not a `BitRange`, which keeps its source's type: a limb list at the
    /// target's widths needs each limb typed at its position's width. The cast lowers to the same
    /// mask, witness and range check.
    fn truncate_limb(&mut self, limb: ValueId, width: usize) -> ValueId {
        self.cast(limb, CastTarget::Int(width))
    }

    /// A witnessed zero of `width` bits, which is a constant and therefore pinned by being one.
    fn zero_limb(&mut self, width: usize) -> ValueId {
        let result = self.fresh();
        self.define_zero(result, width);
        result
    }

    /// The limbs of a wide value as **pure** values, cut to the shape a `to_bits` target needs.
    ///
    /// For narrowing out of the representation to a target wider than the field carries, where
    /// [`Self::recombine`] cannot serve: it sums field elements, and a sum reaching `2^to_bits`
    /// wraps once the target passes the modulus. [`Self::recombine_pure`] reads these back with
    /// integer arithmetic at the target's width.
    fn pure_limbs_at(&mut self, value: ValueId, from_bits: usize, to_bits: usize) -> Vec<ValueId> {
        assert!(
            to_bits <= from_bits,
            "ICE: an int{from_bits} held as limbs was widened to an int{to_bits} on the way out of \
             the representation"
        );
        let witnessed = self.wide_width(value).is_some();
        let source_widths = limb_widths(from_bits, self.limb_bits());
        let limbs = self.limbs(value);

        limb_widths(to_bits, self.limb_bits())
            .into_iter()
            .enumerate()
            .map(|(index, width)| {
                let limb = limbs[index];
                let pure = if witnessed {
                    self.cast(limb, CastTarget::ValueOf)
                } else {
                    limb
                };
                if width < source_widths[index] {
                    self.cast(pure, CastTarget::Int(width))
                } else {
                    pure
                }
            })
            .collect()
    }

    /// The limbs of `value` read at `to_bits`, whatever the two widths are.
    fn relimb(&mut self, value: ValueId, from_bits: usize, to_bits: usize) -> Vec<ValueId> {
        let limb_bits = self.limb_bits();
        let source_widths = limb_widths(from_bits, limb_bits);
        let source = match self.wide_width(value) {
            Some(_) => self.limbs(value),
            None => self.decompose(value, from_bits),
        };

        limb_widths(to_bits, limb_bits)
            .into_iter()
            .enumerate()
            .map(|(index, width)| match source.get(index) {
                None => self.zero_limb(width),
                Some(limb) => match width.cmp(&source_widths[index]) {
                    std::cmp::Ordering::Equal => *limb,
                    std::cmp::Ordering::Greater => self.cast(*limb, CastTarget::Int(width)),
                    std::cmp::Ordering::Less => self.truncate_limb(*limb, width),
                },
            })
            .collect()
    }
}

/// The rewriter is an emitter, so the shared lowering helpers can build into it. Everything it
/// emits goes through [`Rewriter::push`], which tracks the constants a cast carries.
impl HLEmitter for Rewriter<'_> {
    fn fresh_value(&mut self) -> ValueId {
        self.fresh()
    }

    fn emit(&mut self, instruction: OpCode) {
        self.push(instruction);
    }

    fn emit_located(&mut self, instruction: LocatedOpCode) {
        // Every instruction the rewriter emits takes the location of the one it replaces.
        self.push(instruction.as_ref().clone());
    }

    fn emit_constant(&mut self, value: Constant) -> ValueId {
        self.ssa.add_const(value)
    }

    fn field(&self) -> FieldConfig {
        self.field
    }
}

/// The operands of an operation one of the gadgets lowers, with the width it is lowered at.
struct Operands {
    /// The width of both operands, which `check_widths` makes the same, and so of every limb list
    /// the gadget splits them into.
    bits: usize,

    /// The left operand as the program names it: the minuend of a difference, the side an ordering
    /// asks is the smaller, and the dividend of a division. One element or limbs, witnessed or
    /// pure.
    lhs: ValueId,

    /// The right operand as the program names it, in the same forms as `lhs`.
    rhs: ValueId,
}

/// Whether `op` is one the gadgets lower even when no value in the function is limbs.
pub(super) fn lowers_a_gadget(op: &OpCode, fti: &FunctionTypeInfo, field: FieldConfig) -> bool {
    chained(op, fti, field).is_some()
        || multiplied(op, fti, field).is_some()
        || divided(op, fti, field).is_some()
        || shifted(op, fti, field).is_some()
}
