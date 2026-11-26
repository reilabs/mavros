use mavros_artifacts::{FieldConfig, FieldId, InputValueOrdered, MavrosField};
use mavros_int_semantics::IntBits;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};

use crate::{
    collections::HashMap,
    compiler::{
        analysis::{flow_analysis::FlowAnalysis, types::Types},
        pass_manager::{AnalysisStore, Pass},
        passes::{shared::limbs::witness_limb_bits, wide_witness_ints::WideWitnessInts},
        ssa::{
            Instruction, Located, SourceLocation, ValueId,
            hlssa::{
                BinaryArithOpKind, CastTarget, Constant, HLSSA, OpCode, Type,
                builder::{HLEmitter, HLSSABuilder},
            },
        },
    },
    forgery::{self, replace_hints},
};

pub(super) fn bn254() -> FieldConfig {
    FieldConfig::bn254()
}

/// The fields the structural audits run on.
pub(super) fn fields() -> [FieldConfig; 2] {
    [bn254(), FieldConfig::goldilocks()]
}

/// Two widths past what one element of `field` holds: one with a narrow top limb, then one with
/// a full one.
pub(super) fn limbed_widths(field: FieldConfig) -> [usize; 2] {
    match field.id() {
        FieldId::Bn254 => [254, 320],
        FieldId::Goldilocks => [100, 64],
    }
}

/// The limb widths the end-to-end tests cut at: the field's default, and a narrowed one at half of
/// it.
pub(super) fn default_and_narrowed_width(field: FieldConfig) -> [usize; 2] {
    let default = witness_limb_bits(field);
    [default, default / 2]
}

/// Every emitted opcode of one function, in order.
pub(super) fn emitted(ssa: &HLSSA) -> Vec<OpCode> {
    let main = ssa.get_unique_entrypoint_id();
    let function = ssa.get_function(main);
    function
        .get_block(function.get_entry_id())
        .get_instructions()
        .cloned()
        .collect()
}

/// `main(lhs: WitnessOf<int(bits)>, rhs: rhs_type) -> WitnessOf<int(bits)> { lhs op rhs }`.
pub(super) fn operation_with(kind: BinaryArithOpKind, bits: usize, rhs_type: Type) -> HLSSA {
    operation_with_in(bn254(), kind, bits, rhs_type)
}

/// [`operation_with`] over `field`.
pub(super) fn operation_with_in(
    field: FieldConfig,
    kind: BinaryArithOpKind,
    bits: usize,
    rhs_type: Type,
) -> HLSSA {
    let mut ssa = HLSSA::with_main_in(field, "main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    let (lhs, rhs, result) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
    let mut builder = HLSSABuilder::new(&mut ssa);
    builder.modify_function(main, |fb| {
        let entry = fb.function.get_entry_id();
        let block = fb.function.get_block_mut(entry);
        block.push_parameter(lhs, Type::witness_of(Type::int(bits)));
        block.push_parameter(rhs, rhs_type);
        fb.function
            .add_return_type(Type::witness_of(Type::int(bits)));
        let mut block = fb.test_block(entry);
        block.emit(OpCode::BinaryArithOp {
            kind,
            result,
            lhs,
            rhs,
        });
        block.terminate_return(vec![result]);
    });
    ssa
}

/// Wrap every instruction of `ssa`'s entry block in a guard on a new witnessed condition, which
/// is returned.
pub(super) fn guard_every_instruction(ssa: &mut HLSSA) -> ValueId {
    let main = ssa.get_unique_entrypoint_id();
    let condition = ssa.fresh_value();
    let entry = ssa.get_function_mut(main).get_entry_mut();
    entry.push_parameter(condition, Type::witness_of(Type::int(1)));
    let instructions: Vec<_> = entry
        .take_instructions()
        .into_iter()
        .map(|op| {
            let guarded = OpCode::Guard {
                condition,
                inner: Box::new(op.as_ref().clone()),
            };
            Located::new(guarded, op.location().clone())
        })
        .collect();
    entry.put_instructions(instructions);
    condition
}

/// Whether the circuit for `lhs kind rhs` at `bits`, cut at limbs of at most `limb_bits`, accepts,
/// with `forge` applied to the witness-spilled program: witness generation finishes and every
/// constraint holds.
pub(super) fn operation_accepts<F: MavrosField>(
    field: FieldConfig,
    limb_bits: usize,
    kind: BinaryArithOpKind,
    num_bits: usize,
    (lhs, rhs): (&BigUint, &BigUint),
    guard: Option<bool>,
    forge: impl FnOnce(&mut HLSSA),
) -> bool {
    let mut ssa = HLSSA::with_main_in(field, "main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    let location = || SourceLocation::synthetic("operation_accepts");

    HLSSABuilder::new(&mut ssa).modify_function(main, |builder| {
        builder.function.add_return_type(Type::field());
        let entry = builder.function.get_entry_id();
        let (lhs, rhs, condition) = {
            let mut emitter = builder.block(entry).with_source_location(location());
            let [lhs, rhs] = [lhs, rhs].map(|pattern| {
                let input = emitter.add_parameter(Type::field());
                let zero = emitter.cast_to(CastTarget::Int(num_bits), input);
                let constant = emitter.int_const(IntBits::from_biguint(num_bits, pattern));
                emitter.uadd(zero, constant)
            });
            if guard.is_none() {
                let result = emitter.bin(kind, lhs, rhs);
                let element = emitter.cast_to(CastTarget::Field, result);
                emitter.terminate_return(vec![element]);
                return;
            }
            let input = emitter.add_parameter(Type::field());
            (lhs, rhs, emitter.cast_to(CastTarget::Int(1), input))
        };

        let mut merged = None;
        let merge = builder.add_block(|editor| {
            merged = Some(editor.add_parameter(Type::int(num_bits)));
        });
        let taken = builder.add_block(|_| {});
        let untaken = builder.add_block(|_| {});
        builder
            .block(entry)
            .terminate_jmp_if(condition, taken, untaken);

        {
            let mut emitter = builder.block(taken).with_source_location(location());
            let result = emitter.bin(kind, lhs, rhs);
            emitter.terminate_jmp(merge, vec![result]);
        }

        {
            let mut emitter = builder.block(untaken).with_source_location(location());
            let zero = emitter.int_const(IntBits::zero(num_bits));
            emitter.terminate_jmp(merge, vec![zero]);
        }

        let mut emitter = builder.block(merge).with_source_location(location());
        let element = emitter.cast_to(
            CastTarget::Field,
            merged.expect("the merge has a parameter"),
        );
        emitter.terminate_return(vec![element]);
    });
    let input = |value: u128| InputValueOrdered::Field(F::from_element(field.constant(value)));
    let mut inputs = vec![input(0), input(0)];
    inputs.extend(guard.map(|on| input(on as u128)));
    forgery::accepts_the_forgery_with::<F>(
        ssa,
        &inputs,
        |driver| driver.set_max_limb_bits(limb_bits),
        forge,
    )
}

/// `v` in `num_bits`-wide two's complement representation.
pub(super) fn twos_complement(num_bits: usize, v: &BigInt) -> BigUint {
    let modulus = BigInt::one() << num_bits;
    (((v % &modulus) + &modulus) % &modulus)
        .to_biguint()
        .expect("reduced into `0..2^bits`")
}

/// `v` cut into limbs at `widths`, the top limb read as signed where `signed`.
pub(super) fn limbs(v: &BigInt, widths: &[usize], signed: bool) -> Vec<BigInt> {
    let unsigned_repr = twos_complement(widths.iter().sum(), v);
    let mut shift = 0;
    let mut limbs: Vec<BigInt> = widths
        .iter()
        .map(|width| {
            let limb = (&unsigned_repr >> shift) & ((BigUint::one() << *width) - 1u8);
            shift += width;
            BigInt::from(limb)
        })
        .collect();
    let top_width = widths[widths.len() - 1];
    let top_limb = limbs.last_mut().expect("a value has a limb");
    if signed && top_limb.bit(top_width as u64 - 1) {
        *top_limb -= BigInt::one() << top_width;
    }
    limbs
}

/// The coefficients of `F = lhs·rhs + addend − minuend` as limbs.
pub(super) fn coefficients(
    lhs: &[BigInt],
    rhs: &[BigInt],
    addend: &[BigInt],
    minuend: &[BigInt],
) -> Vec<BigInt> {
    let mut f = vec![BigInt::zero(); lhs.len() + rhs.len() - 1];
    for (i, l) in lhs.iter().enumerate() {
        for (j, r) in rhs.iter().enumerate() {
            f[i + j] += l * r;
        }
    }
    for (j, limb) in addend.iter().enumerate() {
        f[j] += limb;
    }
    for (j, limb) in minuend.iter().enumerate() {
        f[j] -= limb;
    }
    f
}

/// The first `committed` coefficients of the `H` with `F(X) = (β − X)·H(X)` over the field, where
/// `f` are the coefficients of `F` and `β = 2^limb_bits`, by synthetic division: `f_0 = β·h_0`,
/// `f_j = β·h_j − h_{j−1}`, `f_d = −h_{d−1}`.
pub(super) fn carries_over_the_field(
    f: &[BigInt],
    limb_bits: usize,
    modulus: &BigUint,
    committed: usize,
) -> Vec<BigUint> {
    let p = BigInt::from(modulus.clone());
    let reduce = |v: BigInt| ((v % &p) + &p) % &p;
    let beta_inv = reduce(BigInt::one() << limb_bits).modpow(&(&p - 2), &p);
    let mut carries = Vec::with_capacity(f.len() - 1);
    let mut previous = BigInt::zero();
    for coefficient in &f[..f.len() - 1] {
        previous = reduce((coefficient + &previous) * &beta_inv);
        carries.push(previous.to_biguint().expect("reduced"));
    }
    assert!(
        reduce(&f[f.len() - 1] + &previous).is_zero(),
        "F(β) is not zero modulo the field, so no carries satisfy every evaluation"
    );
    assert!(
        carries[committed..].iter().all(Zero::is_zero),
        "the identity reaches past its committed carries"
    );
    carries.truncate(committed);
    assert!(
        carries
            .iter()
            .any(|carry| carry.bits() as usize > limb_bits + 8),
        "the forged carries are small, so the forgery tests nothing"
    );
    carries
}

/// What the carry chain of an operation witnesses.
pub(super) struct IdentityWitnesses {
    /// An unsigned product's answer limbs.
    pub(super) answer_limbs: Vec<ValueId>,
    pub(super) carries: Vec<ValueId>,
}

/// [`IdentityWitnesses`] of `ssa` at limb `limb_bits`.
pub(super) fn identity_witnesses(ssa: &HLSSA, limb_bits: usize) -> IdentityWitnesses {
    use BinaryArithOpKind::{SAdd, UAdd, UDiv, UMul, USub};
    let definitions = definitions(ssa);
    let field_cast_of = |value: ValueId| match definitions.get(&value) {
        Some(OpCode::Cast {
            value,
            target: CastTarget::Field,
            ..
        }) => Some(*value),
        _ => None,
    };
    let int_cast_of = |value: ValueId| match definitions.get(&value) {
        Some(OpCode::Cast {
            value,
            target: CastTarget::Int(_),
            ..
        }) => Some(*value),
        _ => None,
    };
    let bin_arith_op_of = |value: ValueId| match definitions.get(&value) {
        Some(OpCode::BinaryArithOp { kind, lhs, rhs, .. }) => Some((*kind, *lhs, *rhs)),
        _ => None,
    };
    let is_radix = |value: ValueId| match ssa.get_const(value).as_deref() {
        Some(Constant::Int(pattern)) => BigUint::from(pattern) == BigUint::one() << limb_bits,
        _ => false,
    };

    let is_carry = |value: ValueId| match bin_arith_op_of(value) {
        Some((USub, lhs, _)) => field_cast_of(lhs)
            .and_then(bin_arith_op_of)
            .is_some_and(|(kind, ..)| kind == SAdd),
        _ => field_cast_of(value)
            .and_then(bin_arith_op_of)
            .is_some_and(|(kind, _, rhs)| kind == UDiv && is_radix(rhs)),
    };
    // A column's running sum truncated to its limb.
    let is_answer_limb = |value: ValueId| {
        field_cast_of(value)
            .and_then(int_cast_of)
            .and_then(bin_arith_op_of)
            .is_some_and(|(kind, ..)| matches!(kind, SAdd | UAdd | USub | UMul))
    };

    let mut witnesses = IdentityWitnesses {
        answer_limbs: Vec::new(),
        carries: Vec::new(),
    };
    for op in forgery::all_ops(ssa) {
        let OpCode::WriteWitness {
            result: Some(result),
            value,
            ..
        } = op
        else {
            continue;
        };
        if is_carry(value) {
            witnesses.carries.push(result);
        } else if is_answer_limb(value) {
            witnesses.answer_limbs.push(result);
        }
    }
    witnesses
}

/// The opcode defining each value of `ssa`. A constant has none.
pub(super) fn definitions(ssa: &HLSSA) -> HashMap<ValueId, OpCode> {
    forgery::all_ops(ssa)
        .flat_map(|op| {
            op.get_results()
                .map(|result| (*result, op.clone()))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Replace the one hint `is_hint` picks, given the opcode defining each value, with the `num_bits`-wide `value`.
pub(super) fn forge_hint(
    ssa: &mut HLSSA,
    num_bits: usize,
    value: &BigUint,
    is_hint: impl Fn(&OpCode, &HashMap<ValueId, OpCode>) -> bool,
) {
    let definitions = definitions(ssa);
    let hints: Vec<ValueId> = forgery::all_ops(ssa)
        .filter(|op| is_hint(op, &definitions))
        .flat_map(|op| op.get_results().copied().collect::<Vec<_>>())
        .collect();
    assert_eq!(hints.len(), 1, "int{num_bits}: one hint to forge");
    let forged = ssa.add_const(Constant::Int(IntBits::from_biguint(num_bits, value)));
    replace_hints(ssa, |op| {
        op.get_results()
            .any(|result| *result == hints[0])
            .then(|| OpCode::Cast {
                result: hints[0],
                value: forged,
                target: CastTarget::Int(num_bits),
            })
    });
}

/// The hint a signed product's answer limbs are cut from: the double-width product cast to `bits`.
pub(super) fn is_wrapped_product(
    bits: usize,
) -> impl Fn(&OpCode, &HashMap<ValueId, OpCode>) -> bool {
    move |op, definitions| match op {
        OpCode::Cast {
            value,
            target: CastTarget::Int(width),
            ..
        } => {
            *width == bits
                && matches!(
                    definitions.get(value),
                    Some(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::UMul,
                        ..
                    })
                )
        }
        _ => false,
    }
}

/// Replace the value of each `WriteWitness` whose result is `writes[i]` with the field element
/// `values[i]`. Asserts every write is found.
pub(super) fn forge_writes(
    ssa: &mut HLSSA,
    field: FieldConfig,
    writes: &[ValueId],
    values: &[BigUint],
) {
    assert_eq!(writes.len(), values.len(), "one forged value per write");
    let forged: Vec<ValueId> = values
        .iter()
        .map(|value| {
            let element = field
                .from_canonical_limbs(&value.to_u64_digits())
                .expect("below the modulus");
            ssa.add_const(Constant::Field(element))
        })
        .collect();
    let replaced = replace_hints(ssa, |op| match *op {
        OpCode::WriteWitness {
            result: Some(result),
            pinned,
            ..
        } => writes
            .iter()
            .position(|write| *write == result)
            .map(|index| OpCode::WriteWitness {
                result: Some(result),
                value: forged[index],
                pinned,
            }),
        _ => None,
    });
    assert_eq!(replaced, writes.len(), "every write forged");
}

pub(super) fn run_pass(ssa: &mut HLSSA) {
    let flow = FlowAnalysis::run(ssa);
    let types = Types::new().run(ssa, &flow);
    let mut store = AnalysisStore::new();
    store.insert_with_deps(flow, vec![]);
    store.insert_with_deps(types, vec![]);
    WideWitnessInts::new(witness_limb_bits(ssa.field())).run(ssa, &store);
}
