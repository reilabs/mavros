pub mod analysis;
pub mod codegen;
pub mod flow_analysis;
pub mod ir;
pub mod llvm_codegen;
pub mod pass_manager;
pub mod passes;
pub mod r1cs_gen;
pub mod ssa;
pub mod ssa_gen;
pub mod untaint_control_flow;
pub mod witness_cast_insertion;
pub mod witness_info;
pub mod witness_type_inference;

pub type Field = ark_bn254::Fr;
