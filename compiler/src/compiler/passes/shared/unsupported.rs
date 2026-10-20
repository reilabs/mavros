//! Refusal for the widths the configured field cannot carry.
//!
//! Witness lowerings pack values into field elements, so each has a precondition on the field's
//! width. All hold on bn254; routing them through one check registers what a narrower field would
//! need.

use std::fmt;

use mavros_artifacts::FieldConfig;

use crate::compiler::passes::shared::limbs::witness_limb_bits;

/// Refuse a lowering that the configured field is too narrow to carry, naming the field.
///
/// `attempted` says what did not fit; the message adds what the field offers.
pub fn unsupported_on_this_field(attempted: fmt::Arguments<'_>, field: FieldConfig) -> ! {
    ice_usr!(
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

    /// The half of the message the callers do not write: the tail names what the field offers, in
    /// the units the caller's half of the message uses.
    #[test]
    #[should_panic(expected = "254-bit modulus and a 64-bit witness limb")]
    fn the_refusal_names_what_bn254_offers() {
        unsupported_on_this_field(
            format_args!("a lowering that does not exist"),
            FieldConfig::bn254(),
        );
    }

    #[test]
    #[should_panic(expected = "64-bit modulus and a 32-bit witness limb")]
    fn the_refusal_names_what_goldilocks_offers() {
        unsupported_on_this_field(
            format_args!("a lowering that does not exist"),
            FieldConfig::goldilocks(),
        );
    }
}
