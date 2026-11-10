use super::{
    limb_config::LimbConfig,
    types::{limb_types, limb_widths},
    utils::multi_cell_int_bits,
};
use crate::{
    collections::HashMap,
    compiler::{
        analysis::types::TypeInfo,
        ssa::{
            BlockId, FunctionId, Instruction, ValueId,
            hlssa::{CastTarget, HLSSA, OpCode, Type},
        },
    },
};

pub(super) fn plan_function(
    ssa: &HLSSA,
    fid: FunctionId,
    reachable: &[BlockId],
    type_info: &TypeInfo,
    limb_config: LimbConfig,
) -> HashMap<ValueId, Vec<ValueId>> {
    let field = limb_config.field;
    let func = ssa.get_function(fid);
    let fti = type_info.get_function(fid);
    let mut value_map: HashMap<ValueId, Vec<ValueId>> = HashMap::default();

    let plan_value = |value: ValueId, ty: &Type, map: &mut HashMap<ValueId, Vec<ValueId>>| {
        let count = limb_types(ty, limb_config).len();
        let limbs = if count == 1 {
            vec![value]
        } else {
            (0..count).map(|_| ssa.fresh_value()).collect()
        };
        map.insert(value, limbs);
    };

    for bid in reachable {
        for (pid, ty) in func.get_block(*bid).get_parameters() {
            plan_value(*pid, ty, &mut value_map);
        }
    }

    for bid in reachable {
        for instr in func.get_block(*bid).get_instructions() {
            for result in instr.get_results() {
                let ty = fti.get_value_type(*result);
                plan_value(*result, ty, &mut value_map);
            }
        }
    }

    // A wide value's expansion must reach the values that merely **carry** it, which their types
    // do not identify.
    //
    // A witness-indexed array read moves its element through a field element: the lowering reads
    // the hint, strips it to the pure domain, casts that to `FieldLimbs`, writes it as a witness,
    // pins it with `constrain_lookup`, and casts back to the element's width. Only the first and
    // last of those are typed as the wide integer. The middle ones are a pure `int`, a `Field` and
    // a `WitnessOf(Field)`, each of which `limb_types` treats as a single value, as it is
    // everywhere else.
    //
    // So the expansion is propagated, to a fixed point, along the instructions that carry a value
    // without reading it. This is sound, not a heuristic: a value the modulus cannot carry has no
    // single injective field element **in either domain**, so its injective image is one element
    // per limb whether stripped or not; and every step below preserves the value.
    let expansion = |map: &HashMap<ValueId, Vec<ValueId>>, value: ValueId| -> usize {
        map.get(&value).map_or(1, Vec::len)
    };
    loop {
        let mut grew = false;
        for bid in reachable {
            for instr in func.get_block(*bid).get_instructions() {
                let (result, count) = match instr {
                    // The injective field image of a wide integer, pure or witnessed. A `Field`
                    // cast is one element, the residue.
                    OpCode::Cast {
                        result,
                        value,
                        target: CastTarget::FieldLimbs,
                    } => {
                        let Some(bits) = fti
                            .get_value_type(*value)
                            .int_width()
                            .filter(|bits| *bits > multi_cell_int_bits(field))
                        else {
                            continue;
                        };
                        (*result, limb_widths(bits, limb_config.limb_bits).len())
                    }
                    // No conversion, so the result stands for whatever the source does.
                    //
                    // `WitnessOf` and `ValueOf` are deliberately **not** here: the rewriter
                    // decomposes and recombines at them explicitly. A strip yields one pure wide
                    // value, as the pure lane holds it whole.
                    OpCode::Cast {
                        result,
                        value,
                        target: CastTarget::Nop,
                    } => (*result, expansion(&value_map, *value)),
                    // Written to a witness column, one per limb.
                    OpCode::WriteWitness {
                        result: Some(result),
                        value,
                        ..
                    } => (*result, expansion(&value_map, *value)),
                    _ => continue,
                };
                if count <= 1 || expansion(&value_map, result) == count {
                    continue;
                }
                let limbs = (0..count).map(|_| ssa.fresh_value()).collect();
                value_map.insert(result, limbs);
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }

    value_map
}
