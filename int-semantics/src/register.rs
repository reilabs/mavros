//! The register of evaluators bound to this crate, and the relation each one must hold.

/// A test that holds an evaluator to its relation, as the file it lives in and its name.
#[derive(Clone, Copy, Debug)]
pub struct Conformance {
    /// Workspace-relative path of the file holding the test.
    pub path: &'static str,

    /// The test function's name.
    pub test: &'static str,
}

/// One evaluator, and everything that ties it to the model.
#[derive(Clone, Copy, Debug)]
pub struct Evaluator {
    /// Its short name, which is also the heading it has in `docs/int-semantics.md`.
    pub tag: &'static str,

    /// What it is, in a few words.
    pub what: &'static str,

    /// The relation it owes the model. Not equality in every case: a speculative folder may decline
    /// where a final one may not, and an abstract interpreter owes soundness rather than a value at
    /// all.
    pub relation: &'static str,

    /// Workspace-relative paths this evaluator lives in.
    ///
    /// These are the files the sweep in `tests/register.rs` exempts, so the list is not merely
    /// documentation: a file named here is one this crate has licensed to define an integer
    /// operation's answer, and every other file in the workspace is not.
    pub files: &'static [&'static str],

    /// The tests that hold it to [`Self::relation`]. Never empty: a relation nothing checks is a
    /// belief, and the register is meant to hold facts.
    pub conformance: &'static [Conformance],
}

/// Every evaluator that must conform, in pipeline order.
pub const EVALUATORS: &[Evaluator] = &[
    Evaluator {
        tag: "const-fold",
        what: "the SCCP lattice's constant folder",
        relation: "refinement: never a value the model does not give, and never a fold of an \
                   input the model rejects",
        files: &["compiler/src/compiler/analysis/click_cooper/lattice.rs"],
        conformance: &[Conformance {
            path: "compiler/src/compiler/analysis/click_cooper/lattice.rs",
            test: "folding_refines_the_reference_model",
        }],
    },
    Evaluator {
        tag: "specializer",
        what: "the specializer's fold of a callee body against a call site",
        relation: "delegation: it evaluates no integer of its own, and routes every fold through \
                   `const-fold`",
        files: &["compiler/src/compiler/passes/specializer.rs"],
        conformance: &[Conformance {
            path: "compiler/src/compiler/analysis/click_cooper/lattice.rs",
            test: "folding_refines_the_reference_model",
        }],
    },
    Evaluator {
        tag: "r1cs-fold",
        what: "the R1CS symbolic executor",
        relation: "exact, and it may not decline: it runs after the guard IR, so `residue` is the \
                   obligation and an input the model leaves unspecified is an ICE",
        files: &["compiler/src/compiler/codegen/hlssa_to_r1cs.rs"],
        conformance: &[
            Conformance {
                path: "compiler/src/compiler/codegen/hlssa_to_r1cs.rs",
                test: "the_r1cs_fold_agrees_with_the_model",
            },
            Conformance {
                path: "compiler/src/compiler/codegen/hlssa_to_r1cs.rs",
                test: "the_unspecified_inputs_are_exactly_the_divmod_ones",
            },
            Conformance {
                path: "compiler/src/compiler/codegen/hlssa_to_r1cs.rs",
                test: "an_out_of_range_shift_amount_masks_to_the_width",
            },
        ],
    },
    Evaluator {
        tag: "cost-model",
        what: "the cost interpreter that drives specialization decisions",
        relation: "delegation to `residue`, answering zero where the model declines to specify \
                   an input at all",
        files: &["compiler/src/compiler/analysis/instrumenter.rs"],
        conformance: &[
            Conformance {
                path: "compiler/src/compiler/analysis/instrumenter.rs",
                test: "the_integer_arm_delegates_with_the_operands_and_reading_it_was_given",
            },
            Conformance {
                path: "compiler/src/compiler/analysis/instrumenter.rs",
                test: "an_out_of_range_shift_amount_masks_rather_than_saturating",
            },
        ],
    },
    Evaluator {
        tag: "vm",
        what: "the bytecode interpreter's integer opcodes",
        relation: "total, and equal to `residue` wherever the model specifies a pattern: no \
                   panic, no process abort, and every answer inside the operand width",
        files: &["vm/src/bytecode.rs"],
        conformance: &[
            Conformance {
                path: "vm/src/bytecode.rs",
                test: "the_int_lane_agrees_with_the_model",
            },
            Conformance {
                path: "vm/src/bytecode.rs",
                test: "the_int128_lane_agrees_with_the_model",
            },
            Conformance {
                path: "vm/src/bytecode.rs",
                test: "the_comparison_opcodes_agree_with_the_model",
            },
            Conformance {
                path: "vm/src/bytecode.rs",
                test: "the_complement_opcode_agrees_with_the_model",
            },
        ],
    },
    Evaluator {
        tag: "llvm",
        what: "the LLVM backend, which reaches WASM",
        relation: "the VM's, read back through LLVM's own constant folder rather than through a \
                   Rust mirror of the lowering's choices",
        files: &["compiler/src/compiler/codegen/llssa_to_llvm.rs"],
        conformance: &[Conformance {
            path: "compiler/src/compiler/codegen/llssa_to_llvm.rs",
            test: "the_emitted_instructions_agree_with_the_model",
        }],
    },
    Evaluator {
        tag: "value-range",
        what: "the interval domain's arithmetic transfer",
        relation: "soundness: every value a model-accepted execution can produce lies in the \
                   range answered. A rejected execution produces no value, so the analysis owes \
                   it nothing",
        files: &["compiler/src/compiler/analysis/value_range_analysis.rs"],
        conformance: &[Conformance {
            path: "compiler/src/compiler/analysis/value_range_analysis.rs",
            test: "the_transfer_is_sound_for_every_accepted_execution",
        }],
    },
    Evaluator {
        tag: "totality",
        what: "the PRE totality oracle, which licenses speculation",
        relation: "an unconditional `true` for a group exactly where the model rejects none of \
                   its inputs",
        files: &["compiler/src/compiler/passes/partial_redundancy_elimination/totality.rs"],
        conformance: &[
            Conformance {
                path: "compiler/src/compiler/passes/partial_redundancy_elimination/totality.rs",
                test: "an_unconditional_verdict_is_given_exactly_where_nothing_can_reject",
            },
            Conformance {
                path: "compiler/src/compiler/passes/partial_redundancy_elimination/totality.rs",
                test: "the_bitwise_operations_are_the_only_total_ones",
            },
        ],
    },
    Evaluator {
        tag: "guard-ir",
        what: "the guard IR, which is what makes a rejection happen at all",
        relation: "it must reject exactly the executions [`crate::eval`] rejects — the other \
                   eight entries are about what an accepted execution answers, and this one is \
                   about which executions are accepted",
        files: &[
            "compiler/src/compiler/passes/shared/overflow_guard.rs",
            "compiler/src/compiler/passes/shared/divmod_guard.rs",
            "compiler/src/compiler/passes/shared/shift_guard.rs",
            "compiler/src/compiler/passes/instruction_lowering/witness_bitwise.rs",
            "compiler/src/compiler/passes/instruction_lowering/witness_integer_arith.rs",
        ],
        conformance: &[
            // The one predicate here that *deletes* a check rather than building one, and so the
            // one whose wrong answer the corpus below cannot see: a discharged check leaves no
            // program to reject.
            Conformance {
                path: "compiler/src/compiler/passes/shared/overflow_guard.rs",
                test: "a_discharged_check_is_one_no_input_could_have_failed",
            },
            Conformance {
                path: "int-semantics/src/corpus.rs",
                test: "every_rejecting_program_is_one_the_model_rejects",
            },
            Conformance {
                path: "int-semantics/src/corpus.rs",
                test: "every_rejection_reason_reaches_the_corpus",
            },
            Conformance {
                path: "int-semantics/tests/generated_corpus.rs",
                test: "the_generated_corpus_matches_the_model",
            },
        ],
    },
];
