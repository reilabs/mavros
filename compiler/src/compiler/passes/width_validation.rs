//! Validation of the integer widths a program asks the compiler to represent.
//!
//! The type system admits integers from a width of 1 through to a maximum of
//! [`MAX_SUPPORTED_INT_BITS`](crate::compiler::ssa::hlssa::MAX_SUPPORTED_INT_BITS), but not every
//! one of those widths can be used in every operation in a well-defined manner. Where a program
//! asks for one that cannot, we refuse it here with a diagnostic rather than compiling something
//! whose meaning cannot be stated.

use mavros_artifacts::FieldConfig;
use mavros_int_semantics::{MAX_LOWERED_SIGNED_BITS, int_bits::HOST_LIMB_BITS};

use crate::{
    collections::HashSet,
    compiler::{
        analysis::types::{FunctionTypeInfo, TypeInfo},
        codegen::bytecode::layout::{DOUBLE_LANE_BITS, SPREAD_MAX_BITS, int_cell_count},
        diagnostic::Diagnostic,
        passes::shared::limbs::{
            narrow_int_bits, two_limb_product_packing_fits, widest_injective_int_bits,
        },
        ssa::{
            SourceLocation, ValueId,
            hlssa::{
                ArithGroup, BinaryArithOpKind, CastTarget, CmpKind, Constant, HLSSA, OpCode, Type,
                TypeExpr, builder::HLBlockEmitter,
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
        funnel.check(op, ssa, types, location, refusals);
    })
}

/// The widths the lowerings build.
struct Funnel {
    /// The widest integer that lives in one field cell and one host word.
    narrow: usize,

    /// The widest integer a lowering reads as two's complement.
    signed: usize,

    /// The widest integer the field carries injectively.
    injective: usize,

    /// The widest integer a sequence element is read at.
    element: usize,

    /// The configured field, for the one bound that is a predicate rather than a width.
    field: FieldConfig,
}

impl Funnel {
    fn new(field: FieldConfig) -> Self {
        let injective = widest_injective_int_bits(field);
        Self {
            narrow: narrow_int_bits(field),
            signed: MAX_LOWERED_SIGNED_BITS,
            injective,
            // Two bounds meet on an element and the narrower of them binds. It has to become a
            // field element, which is the field's question; and the VM's lookup tape addresses it
            // by a tag naming a cell count, of which it has one for a cell and one for a pair, so
            // the widest element it can name is the double lane's. On bn254 the tape binds.
            element: injective.min(DOUBLE_LANE_BITS),
            field,
        }
    }

    /// Refuse `op` where the lowering it reaches does not exist.
    ///
    /// `ssa` is passed to allow the stopgap solution for left-shifts by literals to function. See
    /// [`Self::shift_left_is_lowered`].
    fn check(
        &self,
        op: &OpCode,
        ssa: &HLSSA,
        types: &FunctionTypeInfo,
        location: &SourceLocation,
        refusals: &mut Vec<Diagnostic>,
    ) {
        // A guarded shift is a shift: the lowering unwraps the guard before it dispatches.
        let (_, op) = HLBlockEmitter::unwrap_guard(op);

        match op {
            OpCode::BinaryArithOp { kind, lhs, rhs, .. } => {
                self.check_arith(*kind, *lhs, *rhs, ssa, types, location, refusals)
            }
            OpCode::Cmp { kind, lhs, rhs, .. } => {
                self.check_compare(*kind, *lhs, *rhs, "comparison", types, location, refusals);
            }
            OpCode::AssertCmp { kind, lhs, rhs } => {
                self.check_compare(*kind, *lhs, *rhs, "assertion", types, location, refusals);
            }
            OpCode::Not { value, .. } => self.check_not(*value, types, location, refusals),
            OpCode::SExt {
                value,
                from_bits,
                to_bits,
                ..
            } => self.check_sext(*value, *from_bits, *to_bits, types, location, refusals),
            OpCode::BitRange { value, .. } => {
                self.check_bit_range(*value, types, location, refusals);
            }
            OpCode::Rangecheck { value, max_bits } => {
                self.check_rangecheck(*value, *max_bits, types, location, refusals);
            }
            OpCode::MkSeq { elem_type, .. } | OpCode::MkRepeated { elem_type, .. } => {
                self.check_element(elem_type, location, refusals);
            }
            OpCode::MkSeqOfBlob { element_type, .. } => {
                self.check_element(element_type, location, refusals);
            }
            // Every other opcode either carries no integer width of its own, or reaches a lowering
            // that is generic in one, or reaches a bound this pass deliberately does not state.
            _ => {}
        }
    }

    /// The arithmetic opcode.
    #[allow(clippy::too_many_arguments)]
    fn check_arith(
        &self,
        kind: BinaryArithOpKind,
        lhs: ValueId,
        rhs: ValueId,
        ssa: &HLSSA,
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

        if !self.signed_reading_is_lowered(kind, bits) {
            refusals.push(self.signed_too_wide(operation, bits, location));
            return;
        }

        // Outside the witness domain an integer is a value in the interpreter and in the compiled
        // WASM, both of which compute every operation here at every width, and the overflow check
        // Noir's semantics ask for is built from constants stated at the operand's own width. Only
        // the amount rule at the end of this function binds there.
        if witnessed {
            if bits > self.narrow {
                refusals.push(self.witnessed_too_wide(operation, bits, location));
                return;
            }

            match kind.group() {
                ArithGroup::And | ArithGroup::Or | ArithGroup::Xor => {
                    if !self.bitwise_is_lowered(bits) {
                        refusals.push(self.bitwise_width(operation, bits, location));
                    }
                }
                ArithGroup::Shl | ArithGroup::Shr => {
                    let amount = literal_amount(ssa, rhs);
                    if !bits.is_power_of_two() {
                        refusals.push(self.shift_width(operation, bits, location));
                    } else if kind.group() == ArithGroup::Shl
                        && !self.shift_left_is_lowered(bits, amount)
                    {
                        refusals
                            .push(self.shift_product_too_wide(operation, bits, amount, location));
                    }
                }
                ArithGroup::Mul => {
                    if !self.multiply_is_lowered(bits) {
                        refusals.push(self.product_too_wide(operation, bits, location));
                    }
                }
                ArithGroup::Add | ArithGroup::Sub | ArithGroup::Div | ArithGroup::Rem => {}
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

    /// A comparison, and the assertion of one, which share a lowering and a pair of bounds.
    #[allow(clippy::too_many_arguments)]
    fn check_compare(
        &self,
        kind: CmpKind,
        lhs: ValueId,
        rhs: ValueId,
        operation: &str,
        types: &FunctionTypeInfo,
        location: &SourceLocation,
        refusals: &mut Vec<Diagnostic>,
    ) {
        let Some(bits) = int_width_of(types, lhs) else {
            return;
        };

        // A signed comparison is an operation whose answer depends on the reading.
        if kind.is_signed() && bits > self.signed {
            refusals.push(self.signed_too_wide(operation, bits, location));
            return;
        }

        // Equality is the one comparison with no width of its own: it is a field subtraction and
        // an inverse, which say nothing about how wide the operands are, and above the field it is
        // the conjunction of the limbs' own equalities. An _ordering_ needs the difference to be
        // range-checked at the operands' width, which is where the single cell binds.
        if matches!(kind, CmpKind::Eq) {
            return;
        }

        if (is_witness(types, lhs) || is_witness(types, rhs)) && bits > self.narrow {
            refusals.push(self.witnessed_too_wide(operation, bits, location));
        }
    }

    /// The complement, which is the one bitwise operation lowered for a pure operand as well.
    fn check_not(
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
                    format!("a complement of an int{bits} value is not supported"),
                    location.clone(),
                )
                .with_label(format!("int{bits} does not fit one field element"))
                .with_note(format!(
                    "a complement is computed as `2^bits - 1 - value` in the field, so the widest operand is the widest integer the field carries injectively, int{}",
                    self.injective
                )),
            );
        }
    }

    /// Sign extension, which is lowered through the field for a pure operand as well.
    fn check_sext(
        &self,
        value: ValueId,
        from_bits: usize,
        to_bits: usize,
        types: &FunctionTypeInfo,
        location: &SourceLocation,
        refusals: &mut Vec<Diagnostic>,
    ) {
        if int_width_of(types, value).is_none() {
            return;
        }

        // The source is what is read as two's complement; the target is a wider integer for the
        // result to be deposited in, and is bounded by the single cell that holds it.
        if from_bits > self.signed {
            refusals.push(self.signed_too_wide("sign extension", from_bits, location));
        } else if to_bits > self.narrow {
            refusals.push(self.sext_too_wide(to_bits, location));
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

    /// A sequence element, which is read off the lookup tape by a tag naming its cell count.
    ///
    /// Two things bound it and [`Funnel::element`] is the narrower. Both backends materialise an
    /// array-lookup element as a single `Field`, so an element whose magnitude the modulus cannot
    /// carry has nothing to be read as; and the VM's tape names an element's extent with a tag —
    /// `ELEM_WORD` for one cell and `ELEM_U128` for two — so an element spanning more cells than
    /// there are tags cannot be addressed at all. The multi-cell representation deliberately does
    /// not reach inside a sequence, and widening the tape is unit 7's work.
    fn check_element(
        &self,
        elem_type: &Type,
        location: &SourceLocation,
        refusals: &mut Vec<Diagnostic>,
    ) {
        let Some(bits) = int_width(elem_type) else {
            return;
        };
        if bits <= self.element {
            return;
        }

        let label = if bits > self.injective {
            format!("int{bits} does not fit one field element")
        } else {
            format!("int{bits} spans more cells than the lookup tape can name")
        };

        refusals.push(
            Diagnostic::error(
                format!("an int{bits} sequence element is not supported"),
                location.clone(),
            )
            .with_label(label)
            .with_note(format!(
                "an element is read off the lookup tape as a single field element addressed by a cell count, so the widest element this field holds is int{}",
                self.element
            )),
        );
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

    /// Whether a signed reading of a `bits`-wide operand reaches a lowering.
    ///
    /// A witness lowering encodes the sign as a place value in one field element, so every signed
    /// operation stops there. The pure lane stops at the same width for a different reason: both
    /// backends have wide bodies for an addition, a subtraction, a multiplication and a left shift,
    /// which are the same map on the bit pattern whatever the reading, but the overflow check Noir
    /// asks of each of those is **not**. So the operation being reading-independent buys the pure
    /// lane nothing here: the check its semantics require is not.
    fn signed_reading_is_lowered(&self, kind: BinaryArithOpKind, bits: usize) -> bool {
        !kind.is_signed() || bits <= self.signed
    }

    /// Whether a witnessed bitwise operation at `bits` has a lowering.
    ///
    /// `lower_binary_bitwise` has four arms: `u1` in plain field arithmetic, one limb decomposed
    /// into half-limbs, two of those, and a fall-through that spreads at the operand's own width.
    /// The two limb arms are keyed on the operand's **type** width rather than on the field, and
    /// the fall-through is bounded by the widest spread the bytecode has an instruction for.
    fn bitwise_is_lowered(&self, bits: usize) -> bool {
        bits <= SPREAD_MAX_BITS || bits == HOST_LIMB_BITS || bits == 2 * HOST_LIMB_BITS
    }

    /// Whether a witnessed multiplication at `bits` has a lowering.
    ///
    /// `lower_unsigned_mul` forms the product in one field element, so `2^(2 * bits)` has to stay
    /// below the modulus for the rangecheck on it to tell an honest product from a residue. Its one
    /// escape is the two-limb schoolbook, keyed on a double limb exactly and carrying its own
    /// packing predicate.
    fn multiply_is_lowered(&self, bits: usize) -> bool {
        2 * bits <= self.injective
            || (bits == 2 * HOST_LIMB_BITS && two_limb_product_packing_fits(self.field, bits))
    }

    /// Whether a witnessed left shift at `bits` by `amount` has a sound lowering.
    ///
    /// `wrap_shifted_product` forms `value * 2^amount` in one field element and range-checks it,
    /// which is meaningful only while `2^(bits + amount)` stays below the modulus.
    ///
    /// **`amount` is an amount that the program states as a literal, and nothing weaker.** Where
    /// the program states one, that is what the shift shifts by and the inequality can be decided
    /// exactly; where it does not, the amount is bounded only by the shift's own semantics, which
    /// cap it at `bits - 1`. This is a stopgap solution to keep some tests compiling, and will no
    /// longer be required in the future once shifts are computed limb-wise.
    fn shift_left_is_lowered(&self, bits: usize, amount: Option<usize>) -> bool {
        bits + self.discarded_bits(bits, amount) <= self.injective
    }

    /// How many bits a left shift at `bits` by `amount` can push past the top.
    fn discarded_bits(&self, bits: usize, amount: Option<usize>) -> usize {
        let worst = bits.saturating_sub(1);
        amount.map_or(worst, |amount| amount.min(worst))
    }

    // THE DIAGNOSTICS
    // --------------------------------------------------------------------------------------------

    /// A witnessed value too wide to be constrained as one field element.
    fn witnessed_too_wide(
        &self,
        operation: &str,
        bits: usize,
        location: &SourceLocation,
    ) -> Diagnostic {
        Diagnostic::error(
            format!("a witnessed int{bits} {operation} is not supported"),
            location.clone(),
        )
        .with_label(format!("int{bits} does not fit one witness cell"))
        .with_note(format!(
            "a witnessed integer is constrained as a single field element, so the widest one this field holds is int{}",
            self.narrow
        ))
        .with_note(
            "the same operation is supported at this width outside the witness domain, where the value is computed rather than constrained",
        )
    }

    /// An operand too wide for a lowering that reads two's complement.
    fn signed_too_wide(
        &self,
        operation: &str,
        bits: usize,
        location: &SourceLocation,
    ) -> Diagnostic {
        Diagnostic::error(
            format!("a signed int{bits} {operation} is not supported"),
            location.clone(),
        )
        .with_label(format!("int{bits} is wider than a signed lowering reads"))
        .with_note(format!(
            "a signed operand is read as two's complement in one integer cell, both where a witness lowering encodes the sign as a place value and where an overflow check tests it, so the widest signed operation is int{}",
            self.signed
        ))
    }

    /// A witnessed bitwise operation at a width between the lowerings that exist.
    fn bitwise_width(&self, operation: &str, bits: usize, location: &SourceLocation) -> Diagnostic {
        Diagnostic::error(
            format!("a witnessed int{bits} {operation} is not supported"),
            location.clone(),
        )
        .with_label(format!("int{bits} has no bitwise decomposition"))
        .with_note(format!(
            "a witnessed bitwise operation is decomposed limb-wise at int{} and int{}, and spread bit-wise at every width up to int{SPREAD_MAX_BITS}",
            HOST_LIMB_BITS,
            2 * HOST_LIMB_BITS
        ))
    }

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

    /// A witnessed shift whose width the amount check cannot bound.
    fn shift_width(&self, operation: &str, bits: usize, location: &SourceLocation) -> Diagnostic {
        Diagnostic::error(
            format!("a witnessed int{bits} {operation} is not supported"),
            location.clone(),
        )
        .with_label(format!("int{bits} is not a power of two"))
        .with_note(
            "a witnessed shift bounds its amount by the low bits of it, which is the bound itself only at a power-of-two width",
        )
    }

    /// A witnessed multiplication whose product the field cannot tell from a residue.
    fn product_too_wide(
        &self,
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
            "a witnessed multiplication is a single field product, so the widest one this field carries is int{}",
            self.injective / 2
        ));

        // The one width above that bound with a lowering of its own, where the field affords it.
        if two_limb_product_packing_fits(self.field, 2 * HOST_LIMB_BITS) {
            diagnostic.with_note(format!(
                "int{} is supported despite being wider, because it splits into two limbs and multiplies them schoolbook",
                2 * HOST_LIMB_BITS
            ))
        } else {
            diagnostic
        }
    }

    /// A witnessed left shift whose shifted product the field cannot carry.
    fn shift_product_too_wide(
        &self,
        operation: &str,
        bits: usize,
        amount: Option<usize>,
        location: &SourceLocation,
    ) -> Diagnostic {
        let discarded = self.discarded_bits(bits, amount);
        let label = match amount {
            Some(amount) if amount == discarded => {
                format!(
                    "shifting an int{bits} value by {amount} reaches 2^{}",
                    bits + discarded
                )
            }
            // An amount at or past the width names no shift the lowering performs, so the width's
            // worst case is what it is held to, the same as an amount the program leaves open.
            _ => format!(
                "shifting an int{bits} value by up to {discarded} reaches 2^{}",
                bits + discarded
            ),
        };

        Diagnostic::error(
            format!("a witnessed int{bits} {operation} is not supported"),
            location.clone(),
        )
        .with_label(label)
        .with_note(format!(
            "a witnessed shift forms `value * 2^amount` in one field element and range-checks it, which this field carries up to 2^{}",
            self.injective
        ))
        .with_note(
            "an amount the program states as a literal is held to that amount; any other is held to the widest the width admits, so that whether a shift compiles is read off the program rather than off what an analysis could prove about it",
        )
    }

    /// A sign extension whose target is too wide for the cell its result is computed in.
    fn sext_too_wide(&self, to_bits: usize, location: &SourceLocation) -> Diagnostic {
        Diagnostic::error(
            format!("a sign extension to int{to_bits} is not supported"),
            location.clone(),
        )
        .with_label(format!("int{to_bits} does not fit one field element"))
        .with_note(format!(
            "a sign extension adds a place value to the operand in one field element, so the widest target is int{}",
            self.narrow
        ))
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
        CastTarget::Field => int_width(source),
        CastTarget::Map(inner) => int_width_into_field(element_type(source)?, inner),
        CastTarget::Int(_)
        | CastTarget::WitnessOf
        | CastTarget::ValueOf
        | CastTarget::Nop
        | CastTarget::ArrayToSlice => None,
    }
}

/// The shift amount `value` states as a literal, or [`None`] where the program does not state one.
///
/// A pattern too wide for a `usize` is no answer: the amount is an index into the bits of a value,
/// so one past the host's addressable range names no shift at all, and the width's worst case is
/// the honest bound for it.
fn literal_amount(ssa: &HLSSA, value: ValueId) -> Option<usize> {
    match ssa.get_const(value).as_deref() {
        Some(Constant::Int(pattern)) => usize::try_from(pattern).ok(),
        _ => None,
    }
}

/// The declared width of `value`'s integer type, or [`None`] where it is not an integer.
fn int_width_of(types: &FunctionTypeInfo, value: ValueId) -> Option<usize> {
    int_width(types.try_get_value_type(value)?)
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

/// The width of an integer type, looking through a witness wrapper.
fn int_width(ty: &Type) -> Option<usize> {
    match &ty.expr {
        TypeExpr::Int(bits) => Some(*bits),
        TypeExpr::WitnessOf(inner) => int_width(inner),
        _ => None,
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

    use crate::compiler::{
        analysis::{flow_analysis::FlowAnalysis, types::Types},
        ssa::{
            SourcePosition, Terminator,
            hlssa::{MAX_SUPPORTED_INT_BITS, SequenceTargetType},
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

    /// A witnessed integer is constrained as one field element, so the threshold is where the
    /// representation stops rather than where the type system does.
    #[test]
    fn a_witnessed_operation_stops_at_one_cell() {
        assert!(!refuses(&witnessed(BinaryArithOpKind::UAdd, narrow())));
        assert!(refuses(&witnessed(BinaryArithOpKind::UAdd, narrow() + 1)));
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

    /// One witnessed operand is what a lowering dispatches on, so it is what the rule reads: an
    /// operation between a wide pure value and a narrow witness still reaches the witness lowering.
    #[test]
    fn one_witnessed_operand_is_enough_to_reach_the_lowering() {
        let wide = Type::int(narrow() + 1);
        let witness = Type::witness_of(Type::int(narrow() + 1));

        assert!(refuses(&binary(
            BinaryArithOpKind::UAdd,
            wide,
            witness.clone()
        )));
        assert!(refuses(&binary(
            BinaryArithOpKind::UAdd,
            witness.clone(),
            Type::int(8)
        )));
    }

    /// The three widths `lower_binary_bitwise` decomposes, and the two bands between them.
    #[test]
    fn a_witnessed_bitwise_operation_is_bounded_at_the_widths_it_decomposes() {
        for bits in [1, SPREAD_MAX_BITS, HOST_LIMB_BITS, 2 * HOST_LIMB_BITS] {
            assert!(
                !refuses(&witnessed(BinaryArithOpKind::Xor, bits)),
                "int{bits} has a bitwise lowering"
            );
        }

        for bits in [SPREAD_MAX_BITS + 1, HOST_LIMB_BITS - 1, HOST_LIMB_BITS + 1] {
            assert!(
                refuses(&witnessed(BinaryArithOpKind::And, bits)),
                "int{bits} has no bitwise lowering"
            );
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

    /// The amount rule is the **last** thing a shift is held to, so a program that also meets a
    /// more fundamental bound hears about that one instead.
    #[test]
    fn a_more_fundamental_refusal_is_reported_ahead_of_the_amount() {
        let wide = binary(
            BinaryArithOpKind::UShl,
            Type::witness_of(Type::int(320)),
            Type::witness_of(Type::int(64)),
        );
        let messages: Vec<String> = funnel(&wide)
            .iter()
            .map(|d| d.message().to_string())
            .collect();

        assert_eq!(
            messages,
            vec!["a witnessed int320 left shift is not supported".to_string()],
            "the width refusal is the one to report, and the only one"
        );
    }

    /// A shift's amount is bounded by the low `log2(bits)` bits of it, which bounds it by `bits`
    /// only where `bits` is a power of two.
    #[test]
    fn a_witnessed_shift_needs_a_power_of_two_width() {
        assert!(!refuses(&witnessed(BinaryArithOpKind::UShl, 32)));
        assert!(refuses(&witnessed(BinaryArithOpKind::UShl, 5)));
        assert!(refuses(&witnessed(BinaryArithOpKind::UShr, 96)));
    }

    /// The shifted product lives in one field element, so a shift by an amount the program leaves
    /// open is decided by the _worst_ amount the width admits.
    #[test]
    fn a_witnessed_left_shift_stops_where_its_product_leaves_the_field() {
        let widest_shift = (widest() + 1) / 2;

        assert!(!refuses(&witnessed(
            BinaryArithOpKind::UShl,
            HOST_LIMB_BITS
        )));
        assert!(refuses(&witnessed(
            BinaryArithOpKind::UShl,
            2 * HOST_LIMB_BITS
        )));

        // Only a power-of-two width reaches this rule at all, so the boundary itself is asked of
        // the predicate rather than of a program: every width between the two is refused before it
        // gets here, and the field's own answer would otherwise be invisible.
        let funnel = Funnel::new(FieldConfig::bn254());
        assert!(funnel.shift_left_is_lowered(widest_shift, None));
        assert!(!funnel.shift_left_is_lowered(widest_shift + 1, None));
    }

    /// An amount the program states as a literal is what the shift shifts by, so the product it
    /// reaches is the one the field is asked about.
    #[test]
    fn a_witnessed_left_shift_by_a_literal_is_held_to_that_amount() {
        let bits = 2 * HOST_LIMB_BITS;
        let widest_amount = widest() - bits;

        assert!(!refuses(&shift_by_literal(bits, 120)));
        assert!(!refuses(&shift_by_literal(bits, widest_amount as u128)));
        assert!(refuses(&shift_by_literal(bits, widest_amount as u128 + 1)));
    }

    /// An amount at or past the width names no shift the lowering performs as the amount check
    /// rejects the program instead, so it bounds the product by nothing and the width's worst case
    /// stands.
    #[test]
    fn a_literal_amount_at_the_width_bounds_nothing() {
        let bits = 2 * HOST_LIMB_BITS;

        assert!(refuses(&shift_by_literal(bits, bits as u128)));
        assert!(refuses(&shift_by_literal(bits, bits as u128 + 7)));
    }

    /// The exemption is the literal and not the width, so a narrow shift keeps compiling at every
    /// amount and a wide one is refused wherever the program does not state an amount.
    #[test]
    fn the_literal_exemption_neither_widens_nor_narrows_the_rest() {
        let bits = 2 * HOST_LIMB_BITS;

        for amount in [0, 1, HOST_LIMB_BITS as u128] {
            assert!(
                !refuses(&shift_by_literal(HOST_LIMB_BITS, amount)),
                "int{HOST_LIMB_BITS} << {amount} has headroom at every amount"
            );
        }

        assert!(refuses(&witnessed(BinaryArithOpKind::UShl, bits)));
    }

    /// A witnessed product lives in one field element too. `int128` is above that bound and
    /// supported anyway, by the two-limb schoolbook.
    #[test]
    fn a_witnessed_multiplication_stops_where_its_product_leaves_the_field() {
        let widest_product = widest() / 2;

        assert!(!refuses(&witnessed(
            BinaryArithOpKind::UMul,
            widest_product
        )));
        assert!(refuses(&witnessed(
            BinaryArithOpKind::UMul,
            widest_product + 1
        )));
        assert!(!refuses(&witnessed(
            BinaryArithOpKind::UMul,
            2 * HOST_LIMB_BITS
        )));
    }

    /// Neither bound reads the operands' range, which is what makes them predictable: the same
    /// program answers the same way however much the analysis could prove about the values.
    #[test]
    fn the_product_bounds_are_stated_on_the_program_and_not_on_a_range() {
        let narrow_operand = Type::witness_of(Type::int(8));
        let wide = Type::witness_of(Type::int(2 * HOST_LIMB_BITS));

        // The amount is a runtime `u8`, so it is below 256 and the lowering has headroom for every
        // value it can take. The program states no amount, so it is refused anyway — the literal
        // is the only thing this rule reads a value off, and a type is not one.
        assert!(refuses(&binary(
            BinaryArithOpKind::UShl,
            wide.clone(),
            narrow_operand
        )));

        // The same for a multiplication one bit above the bound, whatever its operands look like.
        let past = Type::witness_of(Type::int(widest() / 2 + 1));
        assert!(refuses(&binary(
            BinaryArithOpKind::UMul,
            past.clone(),
            past
        )));
    }

    /// A guarded shift is a shift. The lowering unwraps the guard before it dispatches, so a rule
    /// that did not would leave every conditional shift unchecked.
    #[test]
    fn a_guard_does_not_hide_the_operation_it_wraps() {
        let operand = Type::witness_of(Type::int(5));
        let ssa = program_with(
            &[operand.clone(), operand, Type::int(1)],
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

    /// The signed frontier binds in both domains and on every operation, where the unsigned bounds
    /// beside it bind in the witness domain alone.
    #[test]
    fn a_signed_reading_stops_at_the_frontier() {
        let past = MAX_LOWERED_SIGNED_BITS + 1;

        assert!(!refuses(&witnessed(
            BinaryArithOpKind::SAdd,
            MAX_LOWERED_SIGNED_BITS
        )));
        assert!(refuses(&witnessed(BinaryArithOpKind::SAdd, past)));

        assert!(!refuses(&pure(
            BinaryArithOpKind::SAdd,
            MAX_LOWERED_SIGNED_BITS
        )));
        assert!(refuses(&pure(BinaryArithOpKind::SAdd, past)));
        assert!(refuses(&pure(BinaryArithOpKind::SMul, past)));
        assert!(refuses(&pure(BinaryArithOpKind::SShl, past)));
        assert!(refuses(&pure(BinaryArithOpKind::SDiv, past)));
        assert!(refuses(&pure(BinaryArithOpKind::SShr, past)));
    }

    /// A signed comparison is one of the operations whose answer depends on the reading, so it
    /// stops at the frontier outside the witness domain as well.
    #[test]
    fn a_signed_comparison_stops_at_the_frontier() {
        let past = MAX_LOWERED_SIGNED_BITS + 1;
        let compare = |kind: CmpKind, operand: Type| {
            program_with(&[operand.clone(), operand], |values, result| OpCode::Cmp {
                kind,
                result,
                lhs: values[0],
                rhs: values[1],
            })
        };

        assert!(refuses(&compare(CmpKind::SLt, Type::int(past))));
        assert!(!refuses(&compare(CmpKind::ULt, Type::int(past))));
        assert!(refuses(&compare(
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

    /// A complement is computed in the field for a pure operand as much as for a witnessed one, so
    /// it is bounded by what the field carries injectively rather than by the witness threshold.
    #[test]
    fn a_complement_is_bounded_by_the_injective_width_in_both_domains() {
        let complement = |bits: usize| {
            program_with(&[Type::int(bits)], |values, result| OpCode::Not {
                result,
                value: values[0],
            })
        };

        assert!(!refuses(&complement(injective())));
        assert!(refuses(&complement(injective() + 1)));
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

    /// A sequence element is bounded by the **lookup tape**, which is narrower than the field.
    ///
    /// Two bounds meet here and the tape's is the one that binds on bn254: an element has to become
    /// a field element, and it has to be addressed by a tag naming its cell count, of which the VM
    /// has one for a cell and one for a pair.
    #[test]
    fn a_sequence_element_is_bounded_by_the_tape_rather_than_by_the_field() {
        let element = |bits: usize| {
            program_with(&[Type::int(bits)], |values, result| OpCode::MkSeq {
                result,
                elems: vec![values[0]],
                seq_type: SequenceTargetType::Array(1),
                elem_type: Type::int(bits),
            })
        };
        let tape = 2 * HOST_LIMB_BITS;

        assert!(!refuses(&element(tape)));
        assert!(refuses(&element(tape + 1)), "the tape has no third tag");

        // And the bound really is the narrower of the two, which is what a field-only bound would
        // have got wrong: this width is one the modulus carries perfectly well.
        assert!(tape < injective());
        assert!(refuses(&element(injective())));
    }

    /// **Three** opcodes make a sequence and each names its element type differently, so the rule
    /// has to meet all three.
    #[test]
    fn every_sequence_constructor_is_checked() {
        let wide = 2 * HOST_LIMB_BITS + 1;

        let from_elements = program_with(&[Type::int(wide)], |values, result| OpCode::MkSeq {
            result,
            elems: vec![values[0]],
            seq_type: SequenceTargetType::Array(1),
            elem_type: Type::int(wide),
        });
        let repeated = program_with(&[Type::int(wide)], |values, result| OpCode::MkRepeated {
            result,
            element: values[0],
            seq_type: SequenceTargetType::Array(2),
            count: 2,
            elem_type: Type::int(wide),
        });
        let from_blob = program_with(&[Type::blob(Type::int(wide), 2)], |values, result| {
            OpCode::MkSeqOfBlob {
                result,
                element_type: Type::int(wide),
                blob: values[0],
            }
        });

        for (what, ssa) in [
            ("MkSeq", from_elements),
            ("MkRepeated", repeated),
            ("MkSeqOfBlob", from_blob),
        ] {
            assert!(refuses(&ssa), "{what} passed a wide element through");
        }
    }

    /// The source of a sign extension is read as two's complement and its target is deposited in
    /// one field element, so the two ends take different bounds.
    #[test]
    fn a_sign_extension_is_bounded_at_both_ends() {
        let sext = |from_bits: usize, to_bits: usize| {
            program_with(&[Type::int(from_bits)], |values, result| OpCode::SExt {
                result,
                value: values[0],
                from_bits,
                to_bits,
            })
        };

        assert!(!refuses(&sext(MAX_LOWERED_SIGNED_BITS, narrow())));
        assert!(refuses(&sext(MAX_LOWERED_SIGNED_BITS + 1, narrow())));
        assert!(refuses(&sext(MAX_LOWERED_SIGNED_BITS, narrow() + 1)));
    }

    /// The refusal has to name the width and the operation, because a program that reaches one of
    /// these has nothing else to go on.
    #[test]
    fn a_capability_refusal_names_the_operation_and_the_width() {
        let refusals = funnel(&witnessed(BinaryArithOpKind::UMul, narrow() + 1));

        assert_eq!(refusals.len(), 1, "{refusals:?}");
        assert_eq!(
            refusals[0].message(),
            format!(
                "a witnessed int{} multiplication is not supported",
                narrow() + 1
            )
        );
        assert_eq!(refusals[0].location(), &location(1));
    }

    /// An assertion is a comparison reached from a second opcode, so it takes the same two bounds
    /// and differs only in what the diagnostic calls it.
    #[test]
    fn an_assertion_is_bounded_like_the_comparison_it_is() {
        let assertion = |kind: CmpKind, operand: Type| {
            program_with(&[operand.clone(), operand], |values, _| OpCode::AssertCmp {
                kind,
                lhs: values[0],
                rhs: values[1],
            })
        };

        let refusals = funnel(&assertion(
            CmpKind::ULt,
            Type::witness_of(Type::int(narrow() + 1)),
        ));
        assert_eq!(refusals.len(), 1, "{refusals:?}");
        assert_eq!(
            refusals[0].message(),
            format!("a witnessed int{} assertion is not supported", narrow() + 1)
        );

        assert!(refuses(&assertion(
            CmpKind::SLt,
            Type::int(MAX_LOWERED_SIGNED_BITS + 1)
        )));
        assert!(!refuses(&assertion(CmpKind::Eq, Type::int(narrow() + 1))));
    }

    /// Compiler-generated code carries one synthetic location for the whole of it, so a rule that
    /// meets the same shape twice there says the same sentence twice about one point in the
    /// program. The two additions below are separated by the subtraction after the sort, as this is
    /// what a `dedup` would miss.
    #[test]
    fn refusals_repeated_at_one_location_are_reported_once() {
        let wide = Type::witness_of(Type::int(narrow() + 1));
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();

        let lhs = ssa.fresh_value();
        let rhs = ssa.fresh_value();
        let entry = ssa.get_function_mut(main).get_entry_mut();
        entry.push_parameter(lhs, wide.clone());
        entry.push_parameter(rhs, wide);

        for kind in [
            BinaryArithOpKind::UAdd,
            BinaryArithOpKind::USub,
            BinaryArithOpKind::UAdd,
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
                format!("a witnessed int{} addition is not supported", narrow() + 1),
                format!(
                    "a witnessed int{} subtraction is not supported",
                    narrow() + 1
                ),
            ]
        );
    }
}
