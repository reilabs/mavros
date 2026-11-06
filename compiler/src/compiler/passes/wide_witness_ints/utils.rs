use mavros_artifacts::FieldConfig;

use crate::compiler::passes::shared::limbs::widest_injective_int_bits;
/// The widest witnessed integer carried as a single field element.
///
/// A wider value has no element to hold it, so it is held as limbs. This is
/// [`widest_injective_int_bits`], so it is field-derived: 253 on bn254, 63 on goldilocks.
///
/// A gadget that needs more headroom than one element leaves does not move this; it asks for a
/// bound locally, as the carry chain asks [`widest_cell_sum_bits`], one bit narrower.
/// `the_representation_threshold_is_where_an_element_stops_being_injective` fails if this moves
/// without its reason moving too.
pub fn multi_cell_int_bits(field: FieldConfig) -> usize {
    widest_injective_int_bits(field)
}
