pub mod analysis;
pub mod codegen;
pub mod located;
pub mod lowering;
pub mod pass_manager;
pub mod passes;
pub mod ssa;
pub mod untaint_control_flow;
pub mod util;

// The compiler middle-end's field value type. This is distinct from the raw
// `mavros_artifacts::Field` (= `ark_bn254::Fr`) used by the VM, the backends, and the serialized
// artifacts; codegen translates between the two at the boundary via `FieldElement::to_ark` /
// `FieldElement::montgomery_limbs`.
// FIELD-ASSUMPTION: L1-alias
pub use mavros_artifacts::FieldElement;
pub type Field = FieldElement;
