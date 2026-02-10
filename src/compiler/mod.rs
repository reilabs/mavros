pub mod analysis;
pub mod codegen;
pub mod constraint_solver;
pub mod flow_analysis;
pub mod ir;
pub mod llvm_codegen;
pub mod monomorphization;
pub mod pass_manager;
pub mod passes;
pub mod r1cs_gen;
pub mod ssa;
pub mod ssa_gen;
pub mod taint_analysis;
pub mod union_find;
pub mod untaint_control_flow;

pub type Field = ark_bn254::Fr;
