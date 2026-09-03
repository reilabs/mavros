//! Refusal for the widths the configured field cannot carry.
//!
//! Nearly every witness lowering packs a value into a single field element, and each of them
//! therefore carries a precondition on the field's width. On bn254 all of them hold with room to
//! spare, but this is not true of every field. Routing those preconditions through one check makes
//! the set of them a register of what a narrower field would need built.

use std::fmt;

use mavros_artifacts::FieldConfig;

use crate::compiler::passes::shared::limbs::witness_limb_bits;

/// Refuse a lowering that the configured field is too narrow to carry, naming the field.
///
/// `attempted` says what was being lowered and which quantity did not fit; the tail added here says
/// what the field actually offers, so the two halves of the mismatch appear in one message.
pub fn unsupported_on_this_field(attempted: fmt::Arguments<'_>, field: FieldConfig) -> ! {
    unimplemented!(
        "{attempted}. The configured field has a {}-bit modulus and a {}-bit witness limb. This \
         is a limit of the lowering on this field rather than an internal invariant — see \
         docs/field-agnosticism.md, Layer 6.",
        field.field_bit_size(),
        witness_limb_bits(field),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The half of the message the callers do not write.
    ///
    /// On the only configured field every caller's predicate holds, so this is the one property of
    /// the funnel that can be checked without a second field: that the tail names what the field
    /// offers, in the units the caller's own half is written in.
    #[test]
    #[should_panic(expected = "254-bit modulus and a 64-bit witness limb")]
    fn the_refusal_names_what_the_field_offers() {
        unsupported_on_this_field(
            format_args!("a lowering that does not exist"),
            FieldConfig::bn254(),
        );
    }
}
