use std::{collections::HashMap, collections::HashSet, path::Path};

use noirc_frontend::hir::Context;
use noirc_frontend::monomorphization::Monomorphizer;
use noirc_frontend::monomorphization::ast::{Definition, Expression, FuncId as AstFuncId};
use noirc_frontend::monomorphization::visitor::visit_expr;
use noirc_frontend::node_interner::FuncId;

use crate::compiler::ssa_gen::LowLevelReplacement;
use crate::driver::Error;

pub enum ReplacementKind {
    /// A blackbox replacement pair: (constrained_fn, unconstrained_fn).
    /// The unconstrained variant is used when the blackbox is called from
    /// unconstrained context; R1CGen never descends into it.
    Single {
        constrained: &'static str,
        unconstrained: &'static str,
    },
    /// Family of replacements dispatched by array size, each carrying both variants.
    ByArraySize(&'static [(&'static str, &'static str, u32)]),
}

pub struct ReplacementSpec {
    lowlevel_name: &'static str,
    kind: ReplacementKind,
}

pub struct ReplacementCrate {
    pub file_name: &'static str,
    pub dep_name: &'static str,
    pub source: &'static str,
    pub replacements: &'static [ReplacementSpec],
}

impl ReplacementCrate {
    fn function_names(&self) -> Vec<&str> {
        self.replacements
            .iter()
            .flat_map(|spec| match &spec.kind {
                ReplacementKind::Single {
                    constrained,
                    unconstrained,
                } => vec![*constrained, *unconstrained],
                ReplacementKind::ByArraySize(entries) => entries
                    .iter()
                    .flat_map(|(c, uc, _)| [*c, *uc])
                    .collect(),
            })
            .collect()
    }

    pub fn lowlevel_names(&self) -> Vec<&str> {
        self.replacements
            .iter()
            .map(|spec| spec.lowlevel_name)
            .collect()
    }
}

/// Scan the monomorphizer's finished functions to find which LowLevel intrinsics are actually used.
pub fn find_needed_lowlevels(monomorphizer: &Monomorphizer) -> HashSet<String> {
    let mut needed = HashSet::new();
    for func in monomorphizer.finished_functions().values() {
        visit_expr(&func.body, &mut |expr| {
            if let Expression::Ident(ident) = expr {
                if let Definition::LowLevel(name) = &ident.definition {
                    needed.insert(name.clone());
                }
            }
            true
        });
    }
    needed
}

pub const REPLACEMENT_CRATES: &[ReplacementCrate] = &[
    ReplacementCrate {
        file_name: "poseidon2_permutation.nr",
        dep_name: "poseidon2_permutation",
        source: include_str!("../../stdlib_replacements/src/poseidon2_permutation.nr"),
        replacements: &[ReplacementSpec {
            lowlevel_name: "poseidon2_permutation",
            kind: ReplacementKind::ByArraySize(&[
                ("t2", "t2_uc", 2),
                ("t3", "t3_uc", 3),
                ("t4", "t4_uc", 4),
                ("t8", "t8_uc", 8),
                ("t12", "t12_uc", 12),
                ("t16", "t16_uc", 16),
            ]),
        }],
    },
    ReplacementCrate {
        file_name: "sha256_compression.nr",
        dep_name: "sha256_compression",
        source: include_str!("../../stdlib_replacements/src/sha256_compression.nr"),
        replacements: &[ReplacementSpec {
            lowlevel_name: "sha256_compression",
            kind: ReplacementKind::Single {
                constrained: "sha256_compression",
                unconstrained: "sha256_compression_uc",
            },
        }],
    },
];

/// Look up named functions from the root module of a crate, returning their metadata.
fn find_functions_in_crate(
    context: &Context,
    crate_id: noirc_frontend::graph::CrateId,
    function_names: &[&str],
) -> Vec<(String, FuncId, noirc_errors::Location, noirc_frontend::Type)> {
    let def_map = context.def_map(&crate_id).unwrap();
    let root_module = &def_map[def_map.root()];
    let mut result = Vec::new();
    for name in function_names {
        if let Some(func_id) = root_module.find_func_with_name(&(*name).into()) {
            let meta = context.def_interner.function_meta(&func_id);
            result.push((name.to_string(), func_id, meta.location, meta.typ.clone()));
        }
    }
    result
}

/// Prepare a replacement crate and look up its functions.
/// Must be called before creating the Monomorphizer (which borrows context.def_interner).
pub fn prepare_replacement_crate(
    context: &mut Context,
    crate_id: noirc_frontend::graph::CrateId,
    replacement: &ReplacementCrate,
) -> Result<Vec<(String, FuncId, noirc_errors::Location, noirc_frontend::Type)>, Error> {
    let impl_crate_id = noirc_driver::prepare_dependency(context, Path::new(replacement.file_name));
    noirc_driver::add_dep(
        context,
        crate_id,
        impl_crate_id,
        replacement.dep_name.parse().unwrap(),
    );
    noirc_driver::check_crate(
        context,
        impl_crate_id,
        &noirc_driver::CompileOptions::default(),
    )
    .map_err(Error::NoirCompilerError)?;

    Ok(find_functions_in_crate(
        context,
        impl_crate_id,
        &replacement.function_names(),
    ))
}

/// Monomorphize replacement functions and register them as lowlevel replacements.
pub fn add_lowlevel_replacements(
    replacement: &ReplacementCrate,
    functions: &[(String, FuncId, noirc_errors::Location, noirc_frontend::Type)],
    monomorphizer: &mut Monomorphizer,
    lowlevel_replacements: &mut HashMap<String, LowLevelReplacement>,
) {
    let functions_by_name: HashMap<
        &str,
        &(String, FuncId, noirc_errors::Location, noirc_frontend::Type),
    > = functions.iter().map(|f| (f.0.as_str(), f)).collect();

    let queue =
        |m: &mut Monomorphizer, name: &str| -> AstFuncId {
            let (_, func_id, location, fn_type) = functions_by_name[name];
            m.queue_function_with_bindings(
                *func_id,
                *location,
                Default::default(),
                fn_type.clone(),
                Vec::new(),
                None,
            )
        };

    for spec in replacement.replacements {
        let lowlevel = match &spec.kind {
            ReplacementKind::Single {
                constrained,
                unconstrained,
            } => LowLevelReplacement::Single {
                constrained: queue(monomorphizer, constrained),
                unconstrained: queue(monomorphizer, unconstrained),
            },
            ReplacementKind::ByArraySize(entries) => {
                let mut c_map: HashMap<u32, AstFuncId> = HashMap::new();
                let mut uc_map: HashMap<u32, AstFuncId> = HashMap::new();
                for (c_name, uc_name, size) in *entries {
                    c_map.insert(*size, queue(monomorphizer, c_name));
                    uc_map.insert(*size, queue(monomorphizer, uc_name));
                }
                LowLevelReplacement::ByArraySize {
                    constrained: c_map,
                    unconstrained: uc_map,
                }
            }
        };
        lowlevel_replacements.insert(spec.lowlevel_name.to_string(), lowlevel);
    }
}
