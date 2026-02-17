use std::path::Path;

use noirc_frontend::hir::Context;
use noirc_frontend::monomorphization::ast::{
    Definition, Expression, FuncId, GlobalId, Program,
};
use noirc_frontend::monomorphization::visitor::visit_ident_mut;
use tracing::info;

pub struct ForeignImplReplacement {
    pub lowlevel_name: String,
    pub noir_source: String,
}

pub fn builtin_replacements() -> Vec<ForeignImplReplacement> {
    vec![ForeignImplReplacement {
        lowlevel_name: "poseidon2_permutation".to_string(),
        noir_source: include_str!("../stdlib_noir_impls/poseidon2_permutation.nr").to_string(),
    }]
}

pub fn apply_replacements(
    program: &mut Program,
    replacements: &[ForeignImplReplacement],
) {
    for replacement in replacements {
        let replacement_program = compile_replacement(replacement);
        let entry_func_id =
            merge_replacement_into_main(program, replacement_program, &replacement.lowlevel_name);
        rewrite_lowlevel_calls(program, &replacement.lowlevel_name, entry_func_id);
        info!(
            message = %"Replaced foreign function",
            lowlevel_name = %replacement.lowlevel_name,
            entry_func_id = ?entry_func_id,
        );
    }
}

fn compile_replacement(replacement: &ForeignImplReplacement) -> Program {
    let mut file_manager = noirc_driver::file_manager_with_stdlib(Path::new(""));
    file_manager.add_file_with_source(Path::new("main.nr"), replacement.noir_source.clone());

    let parsed_files = nargo::parse_all(&file_manager);
    let mut context = Context::from_ref_file_manager(&file_manager, &parsed_files);
    let crate_id = noirc_driver::prepare_crate(&mut context, Path::new("main.nr"));

    noirc_driver::check_crate(&mut context, crate_id, &Default::default())
        .expect("Failed to compile replacement Noir code");

    let main = context
        .get_main_function(&crate_id)
        .expect("Replacement project has no main function");

    noirc_frontend::monomorphization::monomorphize(main, &mut context.def_interner, false)
        .expect("Failed to monomorphize replacement")
}

fn merge_replacement_into_main(
    main_program: &mut Program,
    replacement_program: Program,
    entry_function_name: &str,
) -> FuncId {
    let func_offset = main_program.functions.len() as u32;

    let global_offset = main_program
        .globals
        .keys()
        .map(|g| g.0)
        .max()
        .map_or(0, |max| max + 1);

    let non_main_functions: Vec<_> = replacement_program
        .functions
        .into_iter()
        .filter(|f| !f.is_entry_point)
        .collect();

    let entry_func_id = non_main_functions
        .iter()
        .find(|f| f.name == entry_function_name)
        .map(|f| FuncId(f.id.0 + func_offset))
        .unwrap_or_else(|| {
            panic!(
                "Entry function '{}' not found in replacement program. Available: {:?}",
                entry_function_name,
                non_main_functions.iter().map(|f| &f.name).collect::<Vec<_>>()
            )
        });

    for mut func in non_main_functions {
        func.id = FuncId(func.id.0 + func_offset);
        offset_func_ids(&mut func.body, func_offset);
        offset_global_ids(&mut func.body, global_offset);
        main_program.functions.push(func);
    }

    for (gid, (name, typ, mut expr)) in replacement_program.globals {
        let new_gid = GlobalId(gid.0 + global_offset);
        offset_global_ids(&mut expr, global_offset);
        main_program.globals.insert(new_gid, (name, typ, expr));
    }

    entry_func_id
}

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
}

fn offset_func_ids(expr: &mut Expression, offset: u32) {
    visit_ident_mut(expr, &mut |ident| {
        if let Definition::Function(id) = &mut ident.definition {
            *id = FuncId(id.0 + offset);
        }
    });
}

fn offset_global_ids(expr: &mut Expression, offset: u32) {
    visit_ident_mut(expr, &mut |ident| {
        if let Definition::Global(id) = &mut ident.definition {
            *id = GlobalId(id.0 + offset);
        }
    });
}
