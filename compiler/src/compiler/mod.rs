pub mod analysis;
pub mod codegen;
pub mod diagnostic;
pub mod located;
pub mod lowering;
pub mod pass_manager;
pub mod passes;
pub mod ssa;
pub mod untaint_control_flow;
pub mod util;

// The compiler middle-end's field value type, spelled `Field` everywhere in this crate. This is
// distinct from a field's raw type (`mavros_artifacts::Field` is bn254's), which the VM, the R1CS
// and the backends are generic over; codegen translates at the boundary with
// `MavrosField::from_element` and `FieldElement::storage_cells`.
// FIELD-ASSUMPTION: L1-alias
pub use mavros_artifacts::FieldElement as Field;
