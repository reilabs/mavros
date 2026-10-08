//! Validation of the integer widths a program asks the compiler to represent.
//!
//! The type system admits integers from a width of 1 through to a maximum of
//! [`MAX_SUPPORTED_INT_BITS`](crate::compiler::ssa::hlssa::MAX_SUPPORTED_INT_BITS), but not every
//! one of those widths can be used in every operation in a well-defined manner. Where a program
//! asks for one that cannot, we refuse it here with a diagnostic rather than compiling something
//! whose meaning cannot be stated.

use mavros_artifacts::FieldConfig;
use mavros_int_semantics::int_bits::HOST_LIMB_BITS;

use crate::{
    collections::HashSet,
    compiler::{
        analysis::types::{FunctionTypeInfo, TypeInfo},
        codegen::bytecode::layout::int_cell_count,
        diagnostic::Diagnostic,
        passes::{
            shared::limbs::{
                single_cell_product_fits, single_cell_shift_fits, single_cell_signed_product_fits,
                two_limb_product_packing_fits, widest_injective_int_bits,
            },
            wide_witness_ints::{
                limb_shift_fits, schoolbook_division_fits, schoolbook_product_fits,
            },
        },
        ssa::{
            SourceLocation, ValueId,
            hlssa::{
                ArithGroup, BinaryArithOpKind, CastTarget, HLSSA, OpCode, Type, TypeExpr,
                builder::HLBlockEmitter,
            },
        },
    },
};

// THE SHARED TRAVERSAL
// ================================================================================================

/// Every refusal `rule` makes over the instructions of `ssa`, in source order.
///
/// The order is by file and then by position, so a program with refusals in several functions reads
/// top to bottom.
fn refusals_from(
    ssa: &HLSSA,
    type_info: &TypeInfo,
    mut rule: impl FnMut(&OpCode, &FunctionTypeInfo, &SourceLocation, &mut Vec<Diagnostic>),
) -> Vec<Diagnostic> {
    let mut refusals = Vec::new();

    for (fid, function) in ssa.iter_functions() {
        let types = type_info.get_function(*fid);
        for (_, block) in function.get_blocks() {
            for (op, location) in block.get_instructions_with_source_locations() {
                rule(op, types, location, &mut refusals);
            }
        }
    }

    refusals.sort_by(|left, right| left.location().cmp(right.location()));

    // Compiler-generated code carries one synthetic location for the whole of it, so a rule that
    // meets the same shape twice there would say the same sentence twice about the same point in
    // the program. Two refusals a reader cannot tell apart are one refusal. `dedup` is no use
    // here: the sort is by location alone, so a repeat is not necessarily adjacent to its twin.
    let mut seen = HashSet::default();
    refusals
        .retain(|refusal| seen.insert((refusal.location().clone(), refusal.message().to_string())));
    refusals
}

// LANGUAGE VALIDATION
// ================================================================================================

/// Every refusal in `ssa` that is about what the program means.
pub(crate) fn language_refusals(ssa: &HLSSA, type_info: &TypeInfo) -> Vec<Diagnostic> {
    let widest = widest_injective_int_bits(ssa.field());

    refusals_from(ssa, type_info, |op, types, location, refusals| {
        let OpCode::Cast { value, target, .. } = op else {
            return;
        };

        let Some(source) = types.try_get_value_type(*value) else {
            return;
        };

        if let Some(bits) = int_width_into_field(source, target)
            && bits > widest
        {
            refusals.push(oversized_cast(bits, widest, location));
        }
    })
}

/// The diagnostic for a cast the field cannot carry.
fn oversized_cast(bits: usize, widest: usize, location: &SourceLocation) -> Diagnostic {
    Diagnostic::error(
        format!("an int{bits} value cannot be cast to Field"),
        location.clone(),
    )
    .with_label(format!("2^{bits} exceeds the field modulus"))
    .with_note(format!(
        "the widest integer this field carries injectively is int{widest}"
    ))
    .with_note(
        "above it, two integers differing by the modulus share an element, so the cast would \
         reduce rather than convert",
    )
}

// CAPABILITY VALIDATION
// ================================================================================================

/// Every shape in `ssa` that reaches a lowering this compiler does not support.
pub(crate) fn capability_refusals(ssa: &HLSSA, type_info: &TypeInfo) -> Vec<Diagnostic> {
    let funnel = Funnel::new(ssa.field());

    refusals_from(ssa, type_info, |op, types, location, refusals| {
        funnel.check(op, types, location, refusals);
    })
}

/// The widths the lowerings build.
struct Funnel {
    /// The widest integer the field carries injectively.
    injective: usize,

    /// The configured field, for the one bound that is a predicate rather than a width.
    field: FieldConfig,
}

impl Funnel {
    fn new(field: FieldConfig) -> Self {
        Self {
            injective: widest_injective_int_bits(field),
            field,
        }
    }

    /// Refuse `op` where the lowering it reaches does not exist.
    fn check(
        &self,
        op: &OpCode,
        types: &FunctionTypeInfo,
        location: &SourceLocation,
        refusals: &mut Vec<Diagnostic>,
    ) {
        // A guarded shift is a shift: the lowering unwraps the guard before it dispatches.
        let (_, op) = HLBlockEmitter::unwrap_guard(op);

        match op {
            OpCode::BinaryArithOp { kind, lhs, rhs, .. } => {
                self.check_arith(*kind, *lhs, *rhs, types, location, refusals)
            }
            // A comparison, and the assertion of one, have no width of their own under either
            // reading. Equality is a field subtraction and an inverse, and past the field the
            // conjunction of the limbs' own equalities. An ordering is the borrow out of the
            // difference, which one element carries while the difference fits it and the carry
            // chain carries past that; a signed one is the unsigned ordering of the operands with
            // their sign bits flipped.
            //
            // Sign extension has none either. It adds a multiple of the source's sign bit to the
            // source in one field element while the target fits one, and past that it fills the
            // target's limbs with that bit.
            OpCode::Cmp { .. }
            | OpCode::AssertCmp { .. }
            | OpCode::Not { .. }
            | OpCode::SExt { .. } => {}
            OpCode::BitRange { value, .. } => {
                self.check_bit_range(*value, types, location, refusals);
            }
            OpCode::Rangecheck { value, max_bits } => {
                self.check_rangecheck(*value, *max_bits, types, location, refusals);
            }
            // Constants have already been folded and pure casts are supported by both
            // backends. Only a witnessed field-to-wide-integer cast needs a canonical limb
            // decomposition that the wide integer pass does not yet implement.
            OpCode::Cast {
                value,
                target: CastTarget::Int(bits),
                ..
            } if *bits > self.injective
                && is_witness(types, *value)
                && types.get_value_type(*value).strip_witness().is_field() =>
            {
                refusals.push(
                    Diagnostic::error(
                        format!("casting a witnessed `Field` to int{bits} is not supported"),
                        location.clone(),
                    )
                    .with_label(format!(
                        "a witnessed `Field` can be cast to an integer of at most {} bits on this field",
                        self.injective
                    ))
                    .with_note("cast to a supported integer width before widening the integer"),
                );
            }
            // Every other opcode either carries no integer width of its own, or reaches a lowering
            // that is generic in one, or reaches a bound this pass deliberately does not state.
            _ => {}
        }
    }

    /// The arithmetic opcode.
    fn check_arith(
        &self,
        kind: BinaryArithOpKind,
        lhs: ValueId,
        rhs: ValueId,
        types: &FunctionTypeInfo,
        location: &SourceLocation,
        refusals: &mut Vec<Diagnostic>,
    ) {
        // Every lowering below reads the operation's width off the left operand, a shift included,
        // as a shift amount is allowed to be narrower than the value it shifts.
        let Some(bits) = int_width_of(types, lhs) else {
            return;
        };
        let witnessed = is_witness(types, lhs) || is_witness(types, rhs);
        let operation = operation_name(kind.group());
        let refusals_before = refusals.len();

        // Outside the witness domain an integer is a value in the interpreter and in the compiled
        // WASM, both of which compute every operation here at every width and under either reading,
        // and the overflow check Noir's semantics ask for is built from constants stated at the
        // operand's own width. Only the amount rule at the end of this function binds there.
        //
        // Inside it, every operation has a lowering at every width but the product, the division
        // and the shift, each of which asks the field its own question below. The bitwise three
        // reach any width the half-limb decomposition's recombination fits an element at, and
        // above the representation threshold they have no cross-limb interaction, so they become
        // the same operation on each limb pair. A sum or difference under either reading is one
        // element while its sum fits one and the carry chain past that, a signed one with its sign
        // bits checked beside it.
        if witnessed {
            match kind.group() {
                ArithGroup::Shl | ArithGroup::Shr => {
                    if !self.shift_is_lowered(kind, bits) {
                        refusals.push(self.shift_too_wide(kind, operation, bits, location));
                    }
                }
                ArithGroup::Mul => {
                    if !self.multiply_is_lowered(kind, bits) {
                        refusals.push(self.product_too_wide(kind, operation, bits, location));
                    }
                }
                ArithGroup::Div | ArithGroup::Rem => {
                    if !self.divide_is_lowered(kind, bits) {
                        refusals.push(self.quotient_too_wide(kind, operation, bits, location));
                    }
                }
                ArithGroup::And
                | ArithGroup::Or
                | ArithGroup::Xor
                | ArithGroup::Add
                | ArithGroup::Sub => {}
            }
        }

        // **Last, and in both domains.** A shift is the one operation whose operands may differ in
        // width, and past the cell lane the opcode cannot read a narrower amount. We check here
        // rather than above because widening the amount does not answer any refusals before it.
        if refusals.len() == refusals_before
            && matches!(kind.group(), ArithGroup::Shl | ArithGroup::Shr)
            && let Some(amount_bits) = int_width_of(types, rhs)
            && int_cell_count(bits) > 1
            && int_cell_count(amount_bits) != int_cell_count(bits)
        {
            refusals.push(self.shift_amount_extent(operation, bits, amount_bits, location));
        }
    }

    /// A bit window, which reconstructs its source from the three pieces it cuts it into.
    ///
    /// The witness lowering witnesses the window and the two pieces either side of it, bounds each
    /// at its own width, and constrains `value == low + window·2^offset + high·2^(offset+width)`.
    /// That sum reaches `2^bits`, so the bound is the widest width one element carries. The hints
    /// are pure arithmetic at the operand's own width.
    fn check_bit_range(
        &self,
        value: ValueId,
        types: &FunctionTypeInfo,
        location: &SourceLocation,
        refusals: &mut Vec<Diagnostic>,
    ) {
        let Some(bits) = int_width_of(types, value) else {
            return;
        };

        if bits > self.injective {
            refusals.push(
                Diagnostic::error(
                    format!("a bit window of an int{bits} value is not supported"),
                    location.clone(),
                )
                .with_label(format!("int{bits} does not fit one field element"))
                .with_note(format!(
                    "a bit window constrains its source against the pieces it is cut into, which is one field element, so the widest source this field carries is int{}",
                    self.injective
                )),
            );
        }
    }

    /// A range check, which is bounded in the witness domain and unbounded outside it.
    ///
    /// The value is a field element by the typing rule, so a bound at or above the widest width the
    /// modulus carries injectively holds for every element there is. A witnessed check at such a
    /// width has nothing to decompose; a pure one is a comparison against `2^max_bits` between two
    /// field elements, which needs no width of its own.
    ///
    /// The lookup a witnessed check becomes is built by the witness lowering, which is downstream.
    fn check_rangecheck(
        &self,
        value: ValueId,
        max_bits: usize,
        types: &FunctionTypeInfo,
        location: &SourceLocation,
        refusals: &mut Vec<Diagnostic>,
    ) {
        if !is_witness(types, value) || max_bits <= self.injective {
            return;
        }

        refusals.push(
            Diagnostic::error(
                format!("a range check to {max_bits} bits is not supported"),
                location.clone(),
            )
            .with_label(format!(
                "every element of this field is below 2^{}",
                self.injective + 1
            ))
            .with_note(format!(
                "a witnessed range check constrains a field element, so a bound above int{} holds for every value it could carry and there is nothing to decompose",
                self.injective
            )),
        );
    }

    /// Whether the single cell takes a witnessed product, quotient or remainder at `bits` under
    /// `kind`'s reading: [`single_cell_product_fits`] for an unsigned one, and
    /// [`single_cell_signed_product_fits`] for a signed one, which has no two-limb escape.
    fn cell_takes_product(&self, kind: BinaryArithOpKind, bits: usize) -> bool {
        if kind.is_signed() {
            single_cell_signed_product_fits(self.field, bits)
        } else {
            single_cell_product_fits(self.field, bits)
        }
    }

    /// Whether a witnessed multiplication at `bits` has a lowering.
    ///
    /// The single cell takes one where [`Self::cell_takes_product`] says so. Past that, a product
    /// goes through the representation's schoolbook, a signed one over the operands' magnitudes,
    /// which asks its own question of the field.
    fn multiply_is_lowered(&self, kind: BinaryArithOpKind, bits: usize) -> bool {
        self.cell_takes_product(kind, bits) || schoolbook_product_fits(self.field, bits)
    }

    /// Whether a witnessed division or remainder at `bits` has a lowering.
    ///
    /// The single cell forms `q·d` in one field element, so it takes the widths whose product it
    /// takes. Past that a division goes through the representation, whose `q·d + r` is a
    /// schoolbook, a signed one over the operands' magnitudes.
    fn divide_is_lowered(&self, kind: BinaryArithOpKind, bits: usize) -> bool {
        self.cell_takes_product(kind, bits) || schoolbook_division_fits(self.field, bits)
    }

    /// Whether a witnessed shift at `bits` has a lowering.
    ///
    /// The single cell takes one where [`single_cell_shift_fits`] says so, at any width, the amount
    /// check being a real `amount < bits`. Past that a shift goes through the representation, which
    /// splits each limb at the amount and so asks its own question of the field: a left shift is
    /// one map on the pattern under either reading, and a signed right shift is the unsigned one
    /// with the sign filled in.
    fn shift_is_lowered(&self, kind: BinaryArithOpKind, bits: usize) -> bool {
        single_cell_shift_fits(self.field, bits, kind.group() == ArithGroup::Shl)
            || limb_shift_fits(self.field)
    }

    // THE DIAGNOSTICS
    // --------------------------------------------------------------------------------------------

    /// A shift whose amount does not cover the cells its opcode reads it from.
    fn shift_amount_extent(
        &self,
        operation: &str,
        bits: usize,
        amount_bits: usize,
        location: &SourceLocation,
    ) -> Diagnostic {
        Diagnostic::error(
            format!("an int{bits} {operation} by an int{amount_bits} amount is not supported"),
            location.clone(),
        )
        .with_label(format!(
            "an int{amount_bits} covers {} frame cell(s) where the operation reads {}",
            int_cell_count(amount_bits),
            int_cell_count(bits)
        ))
        .with_note(
            "an integer opcode addresses both operands at the result's own cell count, so an amount laid out in fewer cells would be read across the slot beside it",
        )
        .with_note(format!(
            "widen the amount to int{bits}, which is the width Noir's elaborator already unifies a shift's operands to"
        ))
    }

    /// A witnessed multiplication whose product the field cannot tell from a residue.
    fn product_too_wide(
        &self,
        kind: BinaryArithOpKind,
        operation: &str,
        bits: usize,
        location: &SourceLocation,
    ) -> Diagnostic {
        let diagnostic = Diagnostic::error(
            format!("a witnessed int{bits} {operation} is not supported"),
            location.clone(),
        )
        .with_label(format!(
            "the product of two int{bits} values reaches 2^{}",
            2 * bits
        ))
        .with_note(format!(
            "a witnessed multiplication is a single field product up to int{}",
            self.injective / 2
        ));

        // The one width above that bound with a lowering of its own, where the field affords it.
        let diagnostic = if !kind.is_signed()
            && two_limb_product_packing_fits(self.field, 2 * HOST_LIMB_BITS)
        {
            diagnostic.with_note(format!(
                "an unsigned int{} is supported despite being wider, because it splits into two limbs and multiplies them schoolbook",
                2 * HOST_LIMB_BITS
            ))
        } else {
            diagnostic
        };

        diagnostic.with_note(
            "past it a multiplication is a schoolbook product of witness limbs, a signed one over the operands' magnitudes, which needs one partial product and the carry it is reduced into to fit a field element, and this field's limb does not leave that room",
        )
    }

    /// A witnessed division or remainder whose `q·d + r` the field cannot tell from a residue.
    ///
    /// Shaped as [`Self::product_too_wide`] is, since the product it is checked by is the one that
    /// does not fit.
    fn quotient_too_wide(
        &self,
        kind: BinaryArithOpKind,
        operation: &str,
        bits: usize,
        location: &SourceLocation,
    ) -> Diagnostic {
        let diagnostic = Diagnostic::error(
            format!("a witnessed int{bits} {operation} is not supported"),
            location.clone(),
        )
        .with_label(format!(
            "a quotient times a divisor of int{bits} reaches 2^{}",
            2 * bits
        ))
        .with_note(format!(
            "a witnessed division is checked as a single field product up to int{}",
            self.injective / 2
        ));

        // The one width above that bound whose product has a lowering of its own.
        let diagnostic = if !kind.is_signed()
            && two_limb_product_packing_fits(self.field, 2 * HOST_LIMB_BITS)
        {
            diagnostic.with_note(format!(
                "an unsigned int{} is supported despite being wider, because its product splits into two limbs and multiplies them schoolbook",
                2 * HOST_LIMB_BITS
            ))
        } else {
            diagnostic
        };

        diagnostic.with_note(
            "past it a division is checked by a schoolbook product of witness limbs, a signed one over the operands' magnitudes, which needs one partial product and the carry it is reduced into to fit a field element, and this field's limb does not leave that room",
        )
    }

    /// A witnessed shift the single cell cannot hold and the limbs cannot take either.
    fn shift_too_wide(
        &self,
        kind: BinaryArithOpKind,
        operation: &str,
        bits: usize,
        location: &SourceLocation,
    ) -> Diagnostic {
        let (label, cell) = if kind.group() == ArithGroup::Shl {
            (
                format!(
                    "shifting an int{bits} value by up to {} reaches 2^{}",
                    bits - 1,
                    2 * bits - 1
                ),
                "a witnessed left shift forms `value * 2^amount` in one field element",
            )
        } else {
            (
                format!(
                    "a quotient times a divisor of int{bits} reaches 2^{}",
                    2 * bits
                ),
                "a witnessed right shift divides by `2^amount` in one field element",
            )
        };
        let diagnostic = Diagnostic::error(
            format!("a witnessed int{bits} {operation} is not supported"),
            location.clone(),
        )
        .with_label(label)
        .with_note(format!(
            "{cell}, which this field carries up to 2^{}",
            self.injective
        ));

        diagnostic.with_note(
            "past it a shift splits each witness limb at the amount, which needs a limb times a power of two below the limb to fit a field element, and this field's limb does not leave that room",
        )
    }
}

// UTILITIES
// ================================================================================================

/// The integer width a cast of `source` carries **into** the field, or [`None`] where it does not
/// cross that way.
///
/// Only the inward direction has a bound, which is why only it is named. Reading a field element
/// back as an integer is defined however wide the target is: an element's magnitude always fits the
/// limbs it is carried in, so a wider target is zero-filled above them and a narrower one
/// truncates, which is the reading `docs/int-semantics.md` gives it.
fn int_width_into_field(source: &Type, target: &CastTarget) -> Option<usize> {
    match target {
        CastTarget::Field => source.int_width(),
        CastTarget::Map(inner) => int_width_into_field(element_type(source)?, inner),
        CastTarget::Int(_)
        | CastTarget::WitnessOf
        | CastTarget::ValueOf
        | CastTarget::Nop
        | CastTarget::ArrayToSlice => None,
    }
}

/// The declared width of `value`'s integer type, or [`None`] where it is not an integer.
fn int_width_of(types: &FunctionTypeInfo, value: ValueId) -> Option<usize> {
    types.try_get_value_type(value)?.int_width()
}

/// Whether `value` is a witness.
fn is_witness(types: &FunctionTypeInfo, value: ValueId) -> bool {
    types
        .try_get_value_type(value)
        .is_some_and(|ty| ty.is_witness_of())
}

/// What a diagnostic calls this operation.
fn operation_name(group: ArithGroup) -> &'static str {
    match group {
        ArithGroup::Add => "addition",
        ArithGroup::Sub => "subtraction",
        ArithGroup::Mul => "multiplication",
        ArithGroup::Div => "division",
        ArithGroup::Rem => "remainder",
        ArithGroup::Shl => "left shift",
        ArithGroup::Shr => "right shift",
        ArithGroup::And => "bitwise and",
        ArithGroup::Or => "bitwise or",
        ArithGroup::Xor => "bitwise xor",
    }
}

/// The element type a mapped cast applies to, looking through a witness wrapper.
fn element_type(ty: &Type) -> Option<&Type> {
    match &ty.expr {
        TypeExpr::Array(element, _) | TypeExpr::Slice(element) | TypeExpr::Blob(element, _) => {
            Some(element)
        }
        TypeExpr::WitnessOf(inner) => element_type(inner),
        _ => None,
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use mavros_artifacts::FieldConfig;

    use crate::compiler::codegen::bytecode::layout::SPREAD_MAX_BITS;
    use crate::compiler::{
        analysis::{flow_analysis::FlowAnalysis, types::Types},
        passes::shared::limbs::narrow_int_bits,
        ssa::{
            SourcePosition, Terminator,
            hlssa::{CmpKind, Constant, MAX_SUPPORTED_INT_BITS},
        },
    };

    /// The widest width the bn254 modulus carries injectively.
    fn widest() -> usize {
        widest_injective_int_bits(FieldConfig::bn254())
    }

    fn location(line: u64) -> SourceLocation {
        SourceLocation::new(
            "cast.nr",
            SourcePosition::new(line, 1),
            SourcePosition::new(line, 20),
        )
    }

    /// `main(value: source) { cast(value, target) }`, one cast at line 1.
    fn program_casting(source: Type, target: CastTarget) -> HLSSA {
        programs_casting(&[(source, target)])
    }

    /// One `main` holding a cast per entry, each on its own line.
    fn programs_casting(casts: &[(Type, CastTarget)]) -> HLSSA {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();

        let mut parameters = Vec::new();
        for (source, _) in casts {
            let parameter = ssa.fresh_value();
            ssa.get_function_mut(main)
                .get_entry_mut()
                .push_parameter(parameter, source.clone());
            parameters.push(parameter);
        }

        for (index, ((_, target), value)) in casts.iter().zip(parameters).enumerate() {
            let result = ssa.fresh_value();
            ssa.get_function_mut(main).get_entry_mut().push_instruction(
                OpCode::Cast {
                    result,
                    value,
                    target: target.clone(),
                }
                .locate(location(index as u64 + 1)),
            );
        }

        ssa.get_function_mut(main)
            .get_entry_mut()
            .set_terminator(Terminator::Return(vec![]));
        ssa
    }

    fn validate(ssa: &HLSSA) -> Vec<Diagnostic> {
        let flow = FlowAnalysis::run(ssa);
        let type_info = Types::new().run(ssa, &flow);
        language_refusals(ssa, &type_info)
    }

    /// The bound is a boundary rather than a number: one width is admitted and the next is not,
    /// and both are read off the field rather than written here.
    #[test]
    fn the_widest_injective_width_casts_and_one_wider_does_not() {
        assert!(validate(&program_casting(Type::int(widest()), CastTarget::Field)).is_empty());

        let refusals = validate(&program_casting(Type::int(widest() + 1), CastTarget::Field));
        assert_eq!(refusals.len(), 1, "{refusals:?}");
    }

    /// A reader has to be able to find the cast and see which width was too wide.
    #[test]
    fn the_diagnostic_names_the_width_and_points_at_the_cast() {
        let bits = widest() + 1;
        let refusals = validate(&program_casting(Type::int(bits), CastTarget::Field));

        assert_eq!(
            refusals[0].message(),
            format!("an int{bits} value cannot be cast to Field")
        );
        assert_eq!(refusals[0].location(), &location(1));
    }

    /// A mapped cast converts every element, so the width that matters is the element's.
    #[test]
    fn a_mapped_cast_is_bounded_at_the_element_width() {
        let array = Type::int(widest() + 1).array_of(4);
        let mapped = CastTarget::Map(Box::new(CastTarget::Field));

        assert_eq!(validate(&program_casting(array, mapped)).len(), 1);
    }

    /// A witnessed integer is an integer of the same width, and the cast is the same cast.
    #[test]
    fn a_witness_wrapper_does_not_hide_the_width() {
        let witnessed = Type::witness_of(Type::int(widest() + 1));

        assert_eq!(
            validate(&program_casting(witnessed, CastTarget::Field)).len(),
            1
        );
    }

    /// The other direction is total at every width: a field element read as an integer takes the
    /// low bits of it, which is defined however wide the target is.
    #[test]
    fn a_cast_from_field_is_not_bounded() {
        let target = CastTarget::Int(crate::compiler::ssa::hlssa::MAX_SUPPORTED_INT_BITS);

        assert!(validate(&program_casting(Type::field(), target)).is_empty());
    }

    /// One compile reports every site, which is the whole reason this collects rather than
    /// refusing at the first one it meets.
    #[test]
    fn every_oversized_cast_is_collected() {
        let wide = Type::int(widest() + 1);
        let refusals = validate(&programs_casting(&[
            (wide.clone(), CastTarget::Field),
            (Type::int(8), CastTarget::Field),
            (wide, CastTarget::Field),
        ]));

        assert_eq!(refusals.len(), 2, "{refusals:?}");
        assert_eq!(refusals[0].location(), &location(1));
        assert_eq!(refusals[1].location(), &location(3));
    }

    /// Refusals come out in source order however the maps behind them are walked. Two blocks, with
    /// the later line in the block that is reached first, so an ordering that follows the walk
    /// cannot pass.
    #[test]
    fn refusals_are_ordered_by_position_rather_than_by_traversal() {
        let wide = Type::int(widest() + 1);
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();

        let entry_value = ssa.fresh_value();
        let entry_result = ssa.fresh_value();
        let second_value = ssa.fresh_value();
        let second_result = ssa.fresh_value();

        let (second, block) = ssa.get_function_mut(main).add_block_mut();
        block.push_parameter(second_value, wide.clone());
        block.push_instruction(
            OpCode::Cast {
                result: second_result,
                value: second_value,
                target: CastTarget::Field,
            }
            .locate(location(2)),
        );
        block.set_terminator(Terminator::Return(vec![]));

        let entry = ssa.get_function_mut(main).get_entry_mut();
        entry.push_parameter(entry_value, wide);
        entry.push_instruction(
            OpCode::Cast {
                result: entry_result,
                value: entry_value,
                target: CastTarget::Field,
            }
            .locate(location(9)),
        );
        entry.set_terminator(Terminator::Jmp(second, vec![entry_value]));

        let refusals = validate(&ssa);
        assert_eq!(refusals.len(), 2, "{refusals:?}");
        assert_eq!(refusals[0].location(), &location(2));
        assert_eq!(refusals[1].location(), &location(9));
    }

    /// A block no edge reaches is typed by nothing, so there is no width to read off a cast in one.
    /// Nothing in the pipeline produces such a block this early; this holds the guard that keeps
    /// that a fact about pass ordering rather than a crash if the ordering ever changes.
    #[test]
    fn a_cast_in_an_unreachable_block_is_passed_over() {
        let mut ssa = program_casting(Type::int(8), CastTarget::Field);
        let main = ssa.get_unique_entrypoint_id();

        let value = ssa.fresh_value();
        let result = ssa.fresh_value();
        let (_, block) = ssa.get_function_mut(main).add_block_mut();
        block.push_parameter(value, Type::int(widest() + 1));
        block.push_instruction(
            OpCode::Cast {
                result,
                value,
                target: CastTarget::Field,
            }
            .locate(location(9)),
        );
        block.set_terminator(Terminator::Return(vec![]));

        assert!(validate(&ssa).is_empty());
    }

    // THE CAPABILITY RULES
    // --------------------------------------------------------------------------------------------

    /// The widest witnessed integer that has a representation, read off the field.
    fn narrow() -> usize {
        narrow_int_bits(FieldConfig::bn254())
    }

    /// The widest integer the configured field carries injectively.
    fn injective() -> usize {
        widest_injective_int_bits(FieldConfig::bn254())
    }

    /// `main(parameters...)` holding one instruction, at line 1.
    fn program_with(
        parameters: &[Type],
        instruction: impl FnOnce(&[ValueId], ValueId) -> OpCode,
    ) -> HLSSA {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();

        let values: Vec<ValueId> = parameters
            .iter()
            .map(|ty| {
                let value = ssa.fresh_value();
                ssa.get_function_mut(main)
                    .get_entry_mut()
                    .push_parameter(value, ty.clone());
                value
            })
            .collect();

        let result = ssa.fresh_value();
        let entry = ssa.get_function_mut(main).get_entry_mut();
        entry.push_instruction(instruction(&values, result).locate(location(1)));
        entry.set_terminator(Terminator::Return(vec![]));
        ssa
    }

    /// One binary operation between two operands of the given types.
    fn binary(kind: BinaryArithOpKind, lhs: Type, rhs: Type) -> HLSSA {
        program_with(&[lhs, rhs], |values, result| OpCode::BinaryArithOp {
            kind,
            result,
            lhs: values[0],
            rhs: values[1],
        })
    }

    /// The same operation between two witnessed operands of one width.
    fn witnessed(kind: BinaryArithOpKind, bits: usize) -> HLSSA {
        let operand = Type::witness_of(Type::int(bits));
        binary(kind, operand.clone(), operand)
    }

    /// The same operation between two pure operands of one width.
    fn pure(kind: BinaryArithOpKind, bits: usize) -> HLSSA {
        binary(kind, Type::int(bits), Type::int(bits))
    }

    /// A witnessed `int{bits}` left shift by an amount the program states as a literal.
    fn shift_by_literal(bits: usize, amount: u128) -> HLSSA {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();

        let value = ssa.fresh_value();
        let literal = ssa.add_const(Constant::int(bits, amount));
        let result = ssa.fresh_value();

        let entry = ssa.get_function_mut(main).get_entry_mut();
        entry.push_parameter(value, Type::witness_of(Type::int(bits)));
        entry.push_instruction(
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::UShl,
                result,
                lhs: value,
                rhs: literal,
            }
            .locate(location(1)),
        );
        entry.set_terminator(Terminator::Return(vec![]));
        ssa
    }

    fn funnel(ssa: &HLSSA) -> Vec<Diagnostic> {
        let flow = FlowAnalysis::run(ssa);
        let type_info = Types::new().run(ssa, &flow);
        capability_refusals(ssa, &type_info)
    }

    /// Whether the funnel refuses a program, which is all most of these tests need.
    fn refuses(ssa: &HLSSA) -> bool {
        !funnel(ssa).is_empty()
    }

    /// The sum, difference and ordering have a lowering at **every** width, under either reading.
    ///
    /// One element carries them while their sum fits it, and the carry chain past that, whether the
    /// operands are one element each or limbs. The widths are the edges of those regimes: the
    /// host word and the width past it; the last width whose sum fits; the one width whose
    /// operands fit and sum does not; the first held as limbs; and the cap.
    #[test]
    fn a_witnessed_sum_difference_or_ordering_has_a_lowering_at_every_width() {
        let widths = [
            HOST_LIMB_BITS,
            HOST_LIMB_BITS + 1,
            narrow(),
            narrow() + 1,
            injective() - 1,
            injective(),
            injective() + 1,
            MAX_SUPPORTED_INT_BITS,
        ];
        let ordering = |kind: CmpKind, bits: usize| {
            let operand = Type::witness_of(Type::int(bits));
            program_with(&[operand.clone(), operand], |values, result| OpCode::Cmp {
                kind,
                result,
                lhs: values[0],
                rhs: values[1],
            })
        };
        for bits in widths {
            for kind in [
                BinaryArithOpKind::UAdd,
                BinaryArithOpKind::USub,
                BinaryArithOpKind::SAdd,
                BinaryArithOpKind::SSub,
            ] {
                assert!(!refuses(&witnessed(kind, bits)), "int{bits} {kind:?}");
            }
            for kind in [CmpKind::ULt, CmpKind::SLt] {
                assert!(!refuses(&ordering(kind, bits)), "int{bits} {kind:?}");
            }
        }
    }

    /// The pure lane is the interpreter and the compiled WASM, which have wide bodies for every
    /// operation. Refusing there would refuse the two backends units 4 and 5 built.
    #[test]
    fn the_same_operation_outside_the_witness_domain_is_not_bounded() {
        assert!(!refuses(&pure(BinaryArithOpKind::UAdd, narrow() + 1)));
        assert!(!refuses(&pure(BinaryArithOpKind::UMul, narrow() + 1)));
        assert!(!refuses(&pure(
            BinaryArithOpKind::UMul,
            MAX_SUPPORTED_INT_BITS
        )));
        assert!(!refuses(&pure(
            BinaryArithOpKind::USub,
            MAX_SUPPORTED_INT_BITS
        )));
        assert!(!refuses(&pure(
            BinaryArithOpKind::UShr,
            MAX_SUPPORTED_INT_BITS
        )));
    }

    /// A witnessed bitwise operation has a lowering at **every** width.
    ///
    /// The widths named here are the ones that used to have none: the two bands between the three
    /// arms `lower_binary_bitwise` was keyed on, and the band between the narrow bound and the
    /// representation threshold. They are kept as the cases rather than a range because each is a
    /// different reason.
    ///
    /// **Bitwise reaches them all because it has no cross-limb interaction.** The other unbounded
    /// operations reach every width for other reasons: equality has no width of its own, and the
    /// sum, difference and ordering have the carry chain, which
    /// [`a_witnessed_sum_difference_or_ordering_has_a_lowering_at_every_width`] holds them to. A new
    /// family should not be admitted without its own such argument.
    #[test]
    fn a_witnessed_bitwise_operation_has_a_lowering_at_every_width() {
        let bands = [
            1,
            SPREAD_MAX_BITS,
            SPREAD_MAX_BITS + 1,
            HOST_LIMB_BITS - 1,
            HOST_LIMB_BITS,
            HOST_LIMB_BITS + 1,
            2 * HOST_LIMB_BITS - 1,
            2 * HOST_LIMB_BITS,
            2 * HOST_LIMB_BITS + 1,
            widest_injective_int_bits(FieldConfig::bn254()),
            MAX_SUPPORTED_INT_BITS,
        ];
        for bits in bands {
            for kind in [
                BinaryArithOpKind::And,
                BinaryArithOpKind::Or,
                BinaryArithOpKind::Xor,
            ] {
                assert!(
                    !refuses(&witnessed(kind, bits)),
                    "int{bits} {kind:?} has a bitwise lowering"
                );
            }
        }
    }

    /// A shift's amount has to cover the cells the opcode reads it from.
    ///
    /// The one operation whose operands the model lets differ, and past the cell lane bytecode
    /// cannot honor that: every integer opcode addresses both operands at the result's own cell
    /// count. Below `check_operand_extents`, which is the backstop for a mismatch the passes
    /// between introduce rather than one a program states.
    #[test]
    fn a_shift_amount_must_cover_the_cells_the_opcode_reads() {
        let shift = |bits: usize, amount_bits: usize| {
            binary(
                BinaryArithOpKind::UShl,
                Type::int(bits),
                Type::int(amount_bits),
            )
        };

        // One cell either side: every width the cell lane holds is read from the same cell, so a
        // narrower amount is reduced there and nothing is read past.
        assert!(!refuses(&shift(64, 8)));
        assert!(!refuses(&shift(32, 32)));

        // Two cells against one, and five against one.
        assert!(refuses(&shift(narrow(), 64)));
        assert!(refuses(&shift(320, 64)));

        // And an amount that covers the same cells is fine at every lane.
        assert!(!refuses(&shift(narrow(), narrow())));
        assert!(!refuses(&shift(65, narrow())));
    }

    /// The amount rule binds a witnessed shift under either reading, and is what a shift of a value
    /// held in more cells than its amount is refused for.
    ///
    /// It is the **last** thing a shift is held to, so a program that also met a more fundamental
    /// bound would hear about that one instead; no witnessed shift meets one on bn254.
    #[test]
    fn the_amount_rule_binds_a_witnessed_shift_under_either_reading() {
        for (kind, operation) in [
            (BinaryArithOpKind::SShr, "right shift"),
            (BinaryArithOpKind::UShl, "left shift"),
        ] {
            let shift = binary(
                kind,
                Type::witness_of(Type::int(320)),
                Type::witness_of(Type::int(64)),
            );
            let messages: Vec<String> = funnel(&shift)
                .iter()
                .map(|d| d.message().to_string())
                .collect();
            assert_eq!(
                messages,
                vec![format!(
                    "an int320 {operation} by an int64 amount is not supported"
                )]
            );
        }
    }

    /// A witnessed shift is lowered at every width under either reading: in one field element
    /// while its product or quotient fits one, and by the representation's limb-wise shift
    /// everywhere else, whatever the width's shape and however the amount is stated. A signed
    /// right shift is the unsigned one with the sign filled in.
    #[test]
    fn a_witnessed_shift_is_lowered_at_every_width() {
        for kind in [
            BinaryArithOpKind::UShl,
            BinaryArithOpKind::SShl,
            BinaryArithOpKind::UShr,
            BinaryArithOpKind::SShr,
        ] {
            for bits in [
                1,
                5,
                HOST_LIMB_BITS,
                96,
                widest() / 2,
                widest() / 2 + 1,
                2 * HOST_LIMB_BITS,
                widest(),
                widest() + 1,
                320,
                MAX_SUPPORTED_INT_BITS,
            ] {
                assert!(
                    !refuses(&witnessed(kind, bits)),
                    "a witnessed int{bits} {kind:?} is lowered"
                );
            }
        }

        // An amount the program states is no different, including one at or past the width, which
        // the lowering rejects rather than this.
        let bits = 2 * HOST_LIMB_BITS;
        for amount in [0, 3, 125, 126, 127, 128, 135] {
            assert!(!refuses(&shift_by_literal(bits, amount)));
        }
    }

    /// Where the single cell stops taking a shift: a `<<` forms `value * 2^n` with `n < bits`, and a
    /// `>>` is the single-cell division by `2^n`, whose product reaches one bit further.
    #[test]
    fn the_single_cell_takes_a_shift_while_its_product_fits() {
        let bn254 = FieldConfig::bn254();
        let widest_shift = (widest() + 1) / 2;
        assert!(single_cell_shift_fits(bn254, widest_shift, true));
        assert!(!single_cell_shift_fits(bn254, widest_shift + 1, true));
        assert!(single_cell_shift_fits(bn254, widest() / 2, false));
        assert!(!single_cell_shift_fits(bn254, widest() / 2 + 1, false));
        assert_eq!(widest_shift, 2 * HOST_LIMB_BITS - 1);
    }

    /// The widths a witnessed product or division meets a change of lowering at on bn254: the
    /// last the signed single cell takes and the one past it, the unsigned two-limb escape, the
    /// first past the host word, the representation threshold either side, and the cap.
    fn product_widths() -> [usize; 7] {
        let widest_product = widest() / 2;
        [
            widest_product,
            widest_product + 1,
            2 * HOST_LIMB_BITS,
            narrow() + 1,
            injective(),
            injective() + 1,
            MAX_SUPPORTED_INT_BITS,
        ]
    }

    /// A witnessed product is lowered at every width under either reading: in one field element
    /// while the product fits one, by the two-limb schoolbook at an unsigned `int128`, and by the
    /// representation's schoolbook everywhere else, a signed one over the operands' magnitudes.
    #[test]
    fn a_witnessed_multiplication_is_lowered_at_every_width() {
        for bits in product_widths() {
            for kind in [BinaryArithOpKind::UMul, BinaryArithOpKind::SMul] {
                assert!(!refuses(&witnessed(kind, bits)), "int{bits} {kind:?}");
            }
        }
    }

    /// A witnessed division and remainder are lowered at every width under either reading, as the
    /// product they are checked by is.
    #[test]
    fn a_witnessed_division_is_lowered_at_every_width() {
        for bits in product_widths() {
            for kind in [
                BinaryArithOpKind::UDiv,
                BinaryArithOpKind::URem,
                BinaryArithOpKind::SDiv,
                BinaryArithOpKind::SRem,
            ] {
                assert!(!refuses(&witnessed(kind, bits)), "int{bits} {kind:?}");
            }
        }
    }

    /// The single cell takes a signed product, quotient or remainder only where `2^(2 · bits)` fits
    /// the modulus: the unsigned two-limb escape at `int128` packs an unsigned product, and a
    /// signed one past the bound goes to the representation.
    #[test]
    fn the_single_cell_takes_a_signed_product_without_the_two_limb_escape() {
        let funnel = Funnel::new(FieldConfig::bn254());
        let widest_product = widest() / 2;
        for kind in [BinaryArithOpKind::SMul, BinaryArithOpKind::SDiv] {
            assert!(funnel.cell_takes_product(kind, widest_product), "{kind:?}");
            assert!(
                !funnel.cell_takes_product(kind, widest_product + 1),
                "{kind:?}"
            );
            assert!(
                !funnel.cell_takes_product(kind, 2 * HOST_LIMB_BITS),
                "{kind:?}"
            );
        }
        assert!(funnel.cell_takes_product(BinaryArithOpKind::UMul, 2 * HOST_LIMB_BITS));
    }

    /// A division's refusal offers the two-limb escape to an unsigned one alone, and names the
    /// schoolbook for both.
    ///
    /// Asked of the diagnostic directly, as no field this crate configures refuses a division.
    #[test]
    fn a_division_refusal_offers_the_two_limb_escape_only_unsigned() {
        let funnel = Funnel::new(FieldConfig::bn254());
        let render = |kind| {
            funnel
                .quotient_too_wide(kind, "division", 200, &location(1))
                .to_string()
        };
        let schoolbook = "past it a division is checked by a schoolbook product";
        let two_limbs = format!(
            "an unsigned int{} is supported despite being wider",
            2 * HOST_LIMB_BITS
        );

        let unsigned = render(BinaryArithOpKind::UDiv);
        assert!(unsigned.contains(schoolbook), "{unsigned}");
        assert!(unsigned.contains(&two_limbs), "{unsigned}");

        let signed = render(BinaryArithOpKind::SDiv);
        assert!(signed.contains(schoolbook), "{signed}");
        assert!(!signed.contains(&two_limbs), "{signed}");
    }

    /// A guarded shift is a shift. The lowering unwraps the guard before it dispatches, so a rule
    /// that did not would leave every conditional shift unchecked.
    #[test]
    fn a_guard_does_not_hide_the_operation_it_wraps() {
        let ssa = program_with(
            &[
                Type::witness_of(Type::int(320)),
                Type::witness_of(Type::int(64)),
                Type::int(1),
            ],
            |values, result| OpCode::Guard {
                condition: values[2],
                inner: Box::new(OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::UShl,
                    result,
                    lhs: values[0],
                    rhs: values[1],
                }),
            },
        );

        assert!(refuses(&ssa));
    }

    /// Every signed operation has a lowering at every width, in either domain.
    #[test]
    fn a_signed_reading_has_a_lowering_at_every_width() {
        for kind in [
            BinaryArithOpKind::SAdd,
            BinaryArithOpKind::SSub,
            BinaryArithOpKind::SMul,
            BinaryArithOpKind::SDiv,
            BinaryArithOpKind::SRem,
            BinaryArithOpKind::SShl,
            BinaryArithOpKind::SShr,
        ] {
            for bits in [
                HOST_LIMB_BITS,
                HOST_LIMB_BITS + 1,
                2 * HOST_LIMB_BITS,
                injective() + 1,
                MAX_SUPPORTED_INT_BITS,
            ] {
                assert!(
                    !refuses(&witnessed(kind, bits)),
                    "a witnessed {kind:?} at {bits}"
                );
                assert!(!refuses(&pure(kind, bits)), "a pure {kind:?} at {bits}");
            }
        }
    }

    /// A signed comparison is the unsigned ordering of the operands with their top bits flipped,
    /// so it reaches every width the unsigned one does, in either domain.
    #[test]
    fn a_signed_comparison_is_lowered_at_every_width() {
        let past = HOST_LIMB_BITS + 1;
        let compare = |kind: CmpKind, operand: Type| {
            program_with(&[operand.clone(), operand], |values, result| OpCode::Cmp {
                kind,
                result,
                lhs: values[0],
                rhs: values[1],
            })
        };

        assert!(!refuses(&compare(CmpKind::SLt, Type::int(past))));
        assert!(!refuses(&compare(
            CmpKind::SLt,
            Type::int(MAX_SUPPORTED_INT_BITS)
        )));
        assert!(!refuses(&compare(
            CmpKind::SLt,
            Type::witness_of(Type::int(past))
        )));
        assert!(!refuses(&compare(
            CmpKind::SLt,
            Type::witness_of(Type::int(MAX_SUPPORTED_INT_BITS))
        )));
        assert!(!refuses(&compare(CmpKind::ULt, Type::int(past))));
        assert!(!refuses(&compare(
            CmpKind::ULt,
            Type::witness_of(Type::int(narrow() + 1))
        )));
    }

    /// Neither direction of the boundary has a capability bound: the packing fills every limb the
    /// element has, so it carries every width the modulus admits.
    ///
    /// What bounds a cast _to_ the field is the modulus, and that is the language rule
    /// [`language_refusals`] one phase earlier — checked by
    /// `an_oversized_cast_to_field_is_refused_with_a_diagnostic`, not here. A cast _from_ it is
    /// bounded by nothing at all, which is why the widest width the type system admits passes.
    #[test]
    fn a_conversion_across_the_field_boundary_is_left_to_the_language_rule() {
        let cast = |source: Type, target: CastTarget| {
            program_with(&[source], |values, result| OpCode::Cast {
                result,
                value: values[0],
                target,
            })
        };

        assert!(!refuses(&cast(Type::int(injective()), CastTarget::Field)));
        assert!(!refuses(&cast(
            Type::field(),
            CastTarget::Int(MAX_SUPPORTED_INT_BITS)
        )));

        // An integer read as a wider integer stays an integer; nothing is packed into a cell.
        assert!(!refuses(&cast(
            Type::int(8),
            CastTarget::Int(MAX_SUPPORTED_INT_BITS)
        )));
    }

    /// A complement has a lowering at every width, in both domains, so no width here refuses one.
    ///
    /// Bit `i` of the answer depends on bit `i` of the operand alone. Outside the witness domain
    /// that is an opcode each backend has at any width; inside it, a value past the representation
    /// threshold is limbs before the lowering sees it and the complement is one per limb, each at
    /// the width that limb actually carries.
    #[test]
    fn a_complement_is_admitted_at_every_width_in_both_domains() {
        let complement = |value_type: Type| {
            program_with(&[value_type], |values, result| OpCode::Not {
                result,
                value: values[0],
            })
        };

        for bits in [8usize, narrow(), narrow() + 1, injective(), injective() + 1] {
            assert!(!refuses(&complement(Type::int(bits))), "pure int{bits}");
            assert!(
                !refuses(&complement(Type::witness_of(Type::int(bits)))),
                "witnessed int{bits}"
            );
        }
    }

    /// A range check is what an integer parameter of `main` reaches, so this is the rule a wide
    /// entry point meets first.
    ///
    /// The bound is the field's, not one cell's: the value is an element, and the decomposition
    /// chunks it at whatever width the element needs. What has nothing to decompose is a bound no
    /// element can exceed.
    #[test]
    fn a_range_check_no_element_can_exceed_is_refused() {
        let rangecheck = |max_bits: usize| {
            program_with(&[Type::witness_of(Type::field())], |values, _| {
                OpCode::Rangecheck {
                    value: values[0],
                    max_bits,
                }
            })
        };

        assert!(!refuses(&rangecheck(narrow() + 1)));
        assert!(!refuses(&rangecheck(injective())));
        assert!(refuses(&rangecheck(injective() + 1)));
    }

    /// Outside the witness domain the same check is a comparison between two field elements, which
    /// no width bounds. Refusing it would refuse a program that compiles.
    #[test]
    fn a_pure_range_check_is_not_bounded() {
        let ssa = program_with(&[Type::field()], |values, _| OpCode::Rangecheck {
            value: values[0],
            max_bits: narrow() + 1,
        });

        assert!(!refuses(&ssa));
    }

    /// A bit window is bounded by the **field**, in both domains.
    ///
    /// The window reconstructs its source from the pieces it cuts it into, in one field element,
    /// so what binds is the widest width an element carries and not the narrower one the lowering
    /// used to mint its mask at.
    #[test]
    fn a_bit_window_is_bounded_by_the_injective_width() {
        let window = |ty: Type| {
            program_with(&[ty], |values, result| OpCode::BitRange {
                result,
                value: values[0],
                offset: 1,
                width: 4,
            })
        };

        assert!(!refuses(&window(Type::int(narrow()))));
        assert!(!refuses(&window(Type::int(narrow() + 1))));
        assert!(!refuses(&window(Type::witness_of(Type::int(injective())))));
        assert!(refuses(&window(Type::int(injective() + 1))));
        assert!(refuses(&window(Type::witness_of(Type::int(
            injective() + 1
        )))));
    }

    /// A sign extension has no bound in either domain. A witnessed one adds a multiple of its sign
    /// bit to the source in one field element while the target fits one, and fills the target's
    /// limbs with that bit past it; a pure one is computed at every width.
    #[test]
    fn a_sign_extension_has_no_bound() {
        let sext = |from_bits: usize, to_bits: usize, witnessed: bool| {
            let source = if witnessed {
                Type::witness_of(Type::int(from_bits))
            } else {
                Type::int(from_bits)
            };
            program_with(&[source], |values, result| OpCode::SExt {
                result,
                value: values[0],
                from_bits,
                to_bits,
            })
        };

        for (from_bits, to_bits) in [
            (HOST_LIMB_BITS, narrow()),
            (HOST_LIMB_BITS + 1, narrow()),
            (HOST_LIMB_BITS, narrow() + 1),
            (narrow(), injective()),
            (narrow(), injective() + 1),
            (injective(), injective() + 1),
            (injective() + 1, MAX_SUPPORTED_INT_BITS),
        ] {
            for witnessed in [true, false] {
                assert!(
                    !refuses(&sext(from_bits, to_bits, witnessed)),
                    "int{from_bits} to int{to_bits}, witnessed: {witnessed}"
                );
            }
        }
    }

    /// The refusal has to name the width and the operation, because a program that reaches one of
    /// these has nothing else to go on. Asked of the diagnostics directly, as no field this crate
    /// configures refuses a witnessed operation.
    #[test]
    fn a_capability_refusal_names_the_operation_and_the_width() {
        let funnel = Funnel::new(FieldConfig::bn254());
        let at = location(1);
        for (refusal, operation) in [
            (
                funnel.product_too_wide(BinaryArithOpKind::SMul, "multiplication", 200, &at),
                "multiplication",
            ),
            (
                funnel.quotient_too_wide(BinaryArithOpKind::SRem, "remainder", 200, &at),
                "remainder",
            ),
            (
                funnel.shift_too_wide(BinaryArithOpKind::SShr, "right shift", 200, &at),
                "right shift",
            ),
        ] {
            assert_eq!(
                refusal.message(),
                format!("a witnessed int200 {operation} is not supported")
            );
            assert_eq!(refusal.location(), &at);
        }
    }

    /// An assertion is a comparison reached from a second opcode, so it has no bound either.
    #[test]
    fn an_assertion_has_no_bound_as_the_comparison_it_is_has_none() {
        let assertion = |kind: CmpKind, operand: Type| {
            program_with(&[operand.clone(), operand], |values, _| OpCode::AssertCmp {
                kind,
                lhs: values[0],
                rhs: values[1],
            })
        };

        for bits in [
            HOST_LIMB_BITS + 1,
            injective(),
            injective() + 1,
            MAX_SUPPORTED_INT_BITS,
        ] {
            assert!(
                !refuses(&assertion(CmpKind::SLt, Type::witness_of(Type::int(bits)))),
                "a witnessed signed int{bits} assertion"
            );
            assert!(!refuses(&assertion(CmpKind::SLt, Type::int(bits))));
        }
        assert!(!refuses(&assertion(
            CmpKind::ULt,
            Type::witness_of(Type::int(narrow() + 1))
        )));
        assert!(!refuses(&assertion(CmpKind::Eq, Type::int(narrow() + 1))));
    }

    /// Compiler-generated code carries one synthetic location for the whole of it, so a rule that
    /// meets the same shape twice there says the same sentence twice about one point in the
    /// program. The two right shifts below are separated by the left shift after the sort, as this
    /// is what a `dedup` would miss.
    #[test]
    fn refusals_repeated_at_one_location_are_reported_once() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();

        let lhs = ssa.fresh_value();
        let rhs = ssa.fresh_value();
        let entry = ssa.get_function_mut(main).get_entry_mut();
        entry.push_parameter(lhs, Type::witness_of(Type::int(320)));
        entry.push_parameter(rhs, Type::witness_of(Type::int(64)));

        for kind in [
            BinaryArithOpKind::UShr,
            BinaryArithOpKind::UShl,
            BinaryArithOpKind::UShr,
        ] {
            let result = ssa.fresh_value();
            ssa.get_function_mut(main).get_entry_mut().push_instruction(
                OpCode::BinaryArithOp {
                    kind,
                    result,
                    lhs,
                    rhs,
                }
                .locate(location(1)),
            );
        }

        ssa.get_function_mut(main)
            .get_entry_mut()
            .set_terminator(Terminator::Return(vec![]));

        let messages: Vec<String> = funnel(&ssa)
            .iter()
            .map(|refusal| refusal.message().to_string())
            .collect();
        assert_eq!(
            messages,
            [
                "an int320 right shift by an int64 amount is not supported",
                "an int320 left shift by an int64 amount is not supported",
            ]
        );
    }
}
