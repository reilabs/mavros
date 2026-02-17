//! Replace foreign/builtin function calls with pure Noir implementations.
//!
//! This module compiles replacement `.nr` files through the Noir pipeline,
//! merges them into the main monomorphized program, and rewrites foreign
//! call sites to point to the replacement functions.

use std::fs;

use noirc_frontend::monomorphization::ast::{
    Definition, Expression, FuncId, GlobalId, Program,
};
use noirc_frontend::monomorphization::visitor::visit_ident_mut;
use tracing::info;

// ---------------------------------------------------------------------------
// Embedded replacement sources
// ---------------------------------------------------------------------------

const POSEIDON2_PERMUTE_NR: &str = include_str!("../stdlib_noir_impls/poseidon2_permute.nr");

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Specifies a single foreign function replacement.
pub struct ForeignImplReplacement {
    /// The LowLevel/Builtin name to replace (e.g., "poseidon2_permutation")
    pub lowlevel_name: String,
    /// The Noir source code for the replacement implementation
    pub noir_source: String,
    /// Name of the entry-point function in the replacement source
    pub entry_function_name: String,
    /// A `fn main(...)` wrapper that calls the entry function.
    /// Required so that Noir's monomorphizer has a root.
    pub main_wrapper: String,
}

/// Returns the built-in set of foreign function replacements shipped with the binary.
pub fn builtin_replacements() -> Vec<ForeignImplReplacement> {
    vec![ForeignImplReplacement {
        lowlevel_name: "poseidon2_permutation".to_string(),
        noir_source: POSEIDON2_PERMUTE_NR.to_string(),
        entry_function_name: "poseidon2_permutation_bn254".to_string(),
        main_wrapper: "fn main(input: [Field; 4]) -> pub [Field; 4] { poseidon2_permutation_bn254(input) }".to_string(),
    }]
}

// ---------------------------------------------------------------------------
// Top-level entry point
// ---------------------------------------------------------------------------

/// Apply all foreign function replacements to a monomorphized program.
///
/// For each replacement:
/// 1. Compile the replacement Noir source as a temporary Nargo project
/// 2. Merge the replacement's monomorphized functions/globals into `program`
/// 3. Rewrite matching `Definition::LowLevel` call sites to `Definition::Function`
pub fn apply_replacements(
    program: &mut Program,
    replacements: &[ForeignImplReplacement],
) {
    for replacement in replacements {
        let replacement_program = compile_replacement(replacement);
        let entry_func_id =
            merge_replacement_into_main(program, replacement_program, &replacement.entry_function_name);
        rewrite_lowlevel_calls(program, &replacement.lowlevel_name, entry_func_id);
        info!(
            message = %"Replaced foreign function",
            lowlevel_name = %replacement.lowlevel_name,
            entry_func_id = ?entry_func_id,
        );
    }
}

// ---------------------------------------------------------------------------
// Compile replacement: create temp Nargo project and run Noir pipeline
// ---------------------------------------------------------------------------

fn compile_replacement(replacement: &ForeignImplReplacement) -> Program {
    // Create a temporary Noir project
    let tmp_dir = tempfile::tempdir().expect("Failed to create temp directory");
    let tmp_path = tmp_dir.path();

    // Write Nargo.toml
    let nargo_toml = r#"[package]
name = "foreign_replacement"
type = "bin"
authors = [""]
compiler_version = ">=0.36.0"

[dependencies]
"#;
    fs::write(tmp_path.join("Nargo.toml"), nargo_toml).unwrap();

    // Write src/main.nr = replacement source + main wrapper
    let src_dir = tmp_path.join("src");
    fs::create_dir(&src_dir).unwrap();
    let main_nr = format!("{}\n{}", replacement.noir_source, replacement.main_wrapper);
    fs::write(src_dir.join("main.nr"), main_nr).unwrap();

    // Run through the Noir pipeline (mirrors Project::new + Driver::run_noir_compiler)
    let toml_path =
        nargo_toml::get_package_manifest(tmp_path).expect("Failed to find Nargo.toml in temp dir");
    let workspace = nargo_toml::resolve_workspace_from_toml(
        &toml_path,
        nargo_toml::PackageSelection::All,
        None,
    )
    .expect("Failed to resolve replacement workspace");

    let mut file_manager = workspace.new_file_manager();
    nargo::insert_all_files_for_workspace_into_file_manager(&workspace, &mut file_manager);
    let parsed_files = nargo::parse_all(&file_manager);

    let package = &workspace.members[0];
    let (mut context, crate_id) =
        nargo::prepare_package(&file_manager, &parsed_files, package);

    noirc_driver::check_crate(
        &mut context,
        crate_id,
        &noirc_driver::CompileOptions {
            deny_warnings: false,
            debug_comptime_in_file: None,
            ..Default::default()
        },
    )
    .expect("Failed to compile replacement Noir code");

    let main = context
        .get_main_function(context.root_crate_id())
        .expect("Replacement project has no main function");

    noirc_frontend::monomorphization::monomorphize(main, &mut context.def_interner, false)
        .expect("Failed to monomorphize replacement")
}

// ---------------------------------------------------------------------------
// Merge replacement program into main program
// ---------------------------------------------------------------------------

fn merge_replacement_into_main(
    main_program: &mut Program,
    replacement_program: Program,
    entry_function_name: &str,
) -> FuncId {
    let func_offset = main_program.functions.len() as u32;

    // Compute global ID offset to avoid collisions
    let main_max_global = main_program
        .globals
        .keys()
        .map(|g| g.0)
        .max()
        .unwrap_or(0);
    let global_offset = if replacement_program.globals.is_empty() {
        0
    } else {
        main_max_global + 1
    };

    // Find the entry function by name
    let entry_func_id = replacement_program
        .functions
        .iter()
        .find(|f| f.name == entry_function_name)
        .map(|f| FuncId(f.id.0 + func_offset))
        .unwrap_or_else(|| {
            panic!(
                "Entry function '{}' not found in replacement program. Available: {:?}",
                entry_function_name,
                replacement_program.functions.iter().map(|f| &f.name).collect::<Vec<_>>()
            )
        });

    // Merge functions (rewrite IDs within each)
    for mut func in replacement_program.functions {
        func.id = FuncId(func.id.0 + func_offset);
        func.is_entry_point = false;
        rewrite_ids_in_expression(&mut func.body, func_offset, global_offset);
        main_program.functions.push(func);
    }

    // Merge globals
    for (gid, (name, typ, mut expr)) in replacement_program.globals {
        let new_gid = GlobalId(gid.0 + global_offset);
        rewrite_ids_in_expression(&mut expr, func_offset, global_offset);
        main_program.globals.insert(new_gid, (name, typ, expr));
    }

    entry_func_id
}

// ---------------------------------------------------------------------------
// Rewrite LowLevel/Builtin call sites in the main program
// ---------------------------------------------------------------------------

fn rewrite_lowlevel_calls(
    program: &mut Program,
    lowlevel_name: &str,
    replacement_func_id: FuncId,
) {
    let rewrite = |expr: &mut Expression| {
        visit_ident_mut(expr, &mut |ident| match &ident.definition {
            Definition::LowLevel(name) | Definition::Builtin(name)
                if name == lowlevel_name =>
            {
                ident.definition = Definition::Function(replacement_func_id);
            }
            _ => {}
        });
    };

    for func in &mut program.functions {
        rewrite(&mut func.body);
    }
    for (_gid, (_name, _typ, expr)) in program.globals.iter_mut() {
        rewrite(expr);
    }
}

// ---------------------------------------------------------------------------
// AST walker: rewrite FuncId and GlobalId references
// ---------------------------------------------------------------------------

fn rewrite_ids_in_expression(
    expr: &mut Expression,
    func_offset: u32,
    global_offset: u32,
) {
    visit_ident_mut(expr, &mut |ident| {
        match &mut ident.definition {
            Definition::Function(id) => {
                *id = FuncId(id.0 + func_offset);
            }
            Definition::Global(id) => {
                *id = GlobalId(id.0 + global_offset);
            }
            _ => {}
        }
    });
}
