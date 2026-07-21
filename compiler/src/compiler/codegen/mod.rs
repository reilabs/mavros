pub mod bytecode;
pub mod constants;
pub mod hlssa_to_r1cs;
pub mod llssa_to_llvm;
pub mod r1cs_compact;

#[derive(Clone, Copy, Debug, Default)]
pub struct CodeGenOptions {
    pub check_constraints: bool,
    pub include_debug_info: bool,
}
