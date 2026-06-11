use std::{
    collections::HashSet,
    fmt::Debug,
    path::{Path, PathBuf},
};

use fm::FileManager;
use itertools::Itertools;
use nargo::{
    package::{Dependency, Package},
    workspace::Workspace,
};
use nargo_toml::PackageSelection::All;
use noirc_driver::stdlib_paths_with_source;
use noirc_frontend::ast::{
    BlockExpression, CallExpression, Expression, ExpressionKind, FunctionKind, Ident, NoirFunction,
    Path as AstPath, PathKind, PathSegment, Pattern, Statement, StatementKind,
};
use noirc_frontend::hir::ParsedFiles;
use noirc_frontend::parser::{ItemKind, ParsedModule};
use noirc_frontend::token::FunctionAttributeKind;

use crate::error::Error;

pub struct Project {
    project_root: PathBuf,
    nargo_workspace: Workspace,
    nargo_file_manager: FileManager,
    nargo_parsed_files: ParsedFiles,
}

/// Mavros stdlib extensions that get injected into the `std/` namespace.
/// These must be added before the embedded stdlib so our modified `std/lib.nr` takes precedence.
const MAVROS_STDLIB_FILES: &[(&str, &str)] = &[
    ("std/lib.nr", include_str!("../../mavros_stdlib/lib.nr")),
    (
        "std/mavros.nr",
        include_str!("../../mavros_stdlib/mavros.nr"),
    ),
    (
        "std/mavros/replacements.nr",
        include_str!("../../mavros_stdlib/replacements.nr"),
    ),
    (
        "std/mavros/replacements/blake3.nr",
        include_str!("../../mavros_stdlib/replacements/blake3.nr"),
    ),
    (
        "std/mavros/replacements/poseidon2_permutation.nr",
        include_str!("../../mavros_stdlib/replacements/poseidon2_permutation.nr"),
    ),
    (
        "std/mavros/replacements/sha256_compression.nr",
        include_str!("../../mavros_stdlib/replacements/sha256_compression.nr"),
    ),
];

/// Foreign stdlib functions that mavros replaces with pure-Noir implementations from
/// `std::mavros::replacements`. The replacement for `#[foreign(name)]` is the function
/// `std::mavros::replacements::<name>::<name>`.
///
/// These would be impractical to implement directly in SSA; they play the role of builtins in
/// upstream Noir. After parsing, each `#[foreign(name)]` shim has its attribute dropped and its
/// empty body rewritten to call the replacement, so the rest of the frontend treats it as an
/// ordinary function: type checking, generic instantiation and the constrained/unconstrained
/// pairing all apply natively, and the mavros pipeline never sees a lowlevel call for it.
const FOREIGN_REPLACEMENTS: &[&str] = &["blake3", "poseidon2_permutation", "sha256_compression"];

/// Rewrite all registered `#[foreign]` shims in the parsed files to call their replacements.
fn replace_foreign_functions(parsed_files: &mut ParsedFiles) {
    let mut replaced: HashSet<&'static str> = HashSet::new();
    for (module, _) in parsed_files.values_mut() {
        replace_foreign_functions_in_module(module, &mut replaced);
    }
    for foreign_name in FOREIGN_REPLACEMENTS {
        assert!(
            replaced.contains(foreign_name),
            "foreign function '{foreign_name}' not found in the parsed sources"
        );
    }
}

fn replace_foreign_functions_in_module(
    module: &mut ParsedModule,
    replaced: &mut HashSet<&'static str>,
) {
    for item in &mut module.items {
        match &mut item.kind {
            ItemKind::Function(function) => replace_foreign_function(function, replaced),
            ItemKind::Submodules(submodule) => {
                replace_foreign_functions_in_module(&mut submodule.contents, replaced);
            }
            _ => {}
        }
    }
}

fn replace_foreign_function(function: &mut NoirFunction, replaced: &mut HashSet<&'static str>) {
    let Some((attribute, _)) = &function.def.attributes.function else {
        return;
    };
    let FunctionAttributeKind::Foreign(attribute_name) = &attribute.kind else {
        return;
    };
    let Some(foreign_name) = FOREIGN_REPLACEMENTS
        .iter()
        .find(|name| *name == attribute_name)
    else {
        return;
    };

    let location = function.def.location;
    let variable = |path: AstPath| Expression {
        kind: ExpressionKind::Variable(path),
        location,
    };

    // The shim's parameters are passed through to the replacement verbatim.
    let arguments = function
        .def
        .parameters
        .iter()
        .map(|param| {
            let Pattern::Identifier(ident) = &param.pattern else {
                panic!("foreign function '{foreign_name}' has a non-identifier parameter pattern")
            };
            variable(AstPath::plain(
                vec![PathSegment {
                    ident: ident.clone(),
                    generics: None,
                    location,
                }],
                location,
            ))
        })
        .collect();

    let segments = ["mavros", "replacements", foreign_name, foreign_name]
        .iter()
        .map(|segment| PathSegment {
            ident: Ident::new(segment.to_string(), location),
            generics: None,
            location,
        })
        .collect();
    let func = variable(AstPath {
        segments,
        kind: PathKind::Crate,
        location,
        kind_location: location,
    });

    let call = Expression {
        kind: ExpressionKind::Call(Box::new(CallExpression {
            func: Box::new(func),
            arguments,
            is_macro_call: false,
        })),
        location,
    };
    function.def.body = BlockExpression {
        statements: vec![Statement {
            kind: StatementKind::Expression(call),
            location,
        }],
    };
    function.def.attributes.function = None;
    function.kind = FunctionKind::Normal;
    replaced.insert(foreign_name);
}

fn parse_workspace(workspace: &Workspace) -> (FileManager, ParsedFiles) {
    // Build file manager manually so we can inject mavros stdlib extensions
    // before the embedded stdlib. Since `add_file_with_source_canonical_path`
    // is a no-op for paths that already exist, our `std/lib.nr` (which adds
    // `pub mod mavros;`) takes precedence over the embedded one.
    let mut file_manager = FileManager::new(&workspace.root_dir);

    // 1. Add mavros stdlib extensions first
    for (path, source) in MAVROS_STDLIB_FILES {
        file_manager.add_file_with_source_canonical_path(Path::new(path), source.to_string());
    }

    // 2. Add the rest of the embedded stdlib (std/lib.nr will be skipped)
    for (path, source) in stdlib_paths_with_source() {
        file_manager.add_file_with_source_canonical_path(Path::new(&path), source);
    }

    // 3. Add workspace files
    nargo::insert_all_files_for_workspace_into_file_manager(workspace, &mut file_manager);
    let mut parsed_files = nargo::parse_all(&file_manager);

    // 4. Rewrite replaced foreign functions to call their pure-Noir implementations
    replace_foreign_functions(&mut parsed_files);
    (file_manager, parsed_files)
}

impl Project {
    pub fn new(project_root: PathBuf) -> Result<Self, Error> {
        // Workspace loading was done based on https://github.com/noir-lang/noir/blob/c3a43abf9be80c6f89560405b65f5241ed67a6b2/tooling/nargo_cli/src/cli/mod.rs#L180
        let toml_path = nargo_toml::get_package_manifest(&project_root)?;

        let nargo_workspace = nargo_toml::resolve_workspace_from_toml(&toml_path, All, None)?;

        let (nargo_file_manager, nargo_parsed_files) = parse_workspace(&nargo_workspace);

        Ok(Self {
            project_root,
            nargo_workspace,
            nargo_file_manager,
            nargo_parsed_files,
        })
    }

    pub fn get_only_crate(&self) -> &Package {
        if self.nargo_workspace.members.len() != 1 {
            panic!(
                "Expected exactly one package in the project, got: {}",
                self.nargo_workspace.members.len()
            );
        }
        &self.nargo_workspace.members[0]
    }

    pub fn file_manager(&self) -> &FileManager {
        &self.nargo_file_manager
    }

    pub fn parsed_files(&self) -> &ParsedFiles {
        &self.nargo_parsed_files
    }
}

impl Debug for Project {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn package_fmt(
            f: &mut std::fmt::Formatter<'_>,
            p: &Package,
            tab: &str,
        ) -> std::fmt::Result {
            writeln!(f, "{}name:       {}", tab, p.name)?;
            writeln!(f, "{}version:    {:?}", tab, p.version)?;
            writeln!(f, "{}type:       {}", tab, p.package_type)?;
            writeln!(f, "{}root_dir:   {:?}", tab, p.root_dir)?;
            writeln!(f, "{}entry_path: {:?}", tab, p.entry_path)?;
            writeln!(f, "{tab}dependencies:")?;

            for (crate_name, dep) in &p.dependencies {
                match dep {
                    Dependency::Local { package } => {
                        writeln!(f, "{tab}  (Local)  Crate: {crate_name}")?;
                        package_fmt(f, package, &format!("  {tab}"))?;
                    }
                    Dependency::Remote { package } => {
                        writeln!(f, "{tab}  (Remote) Crate: {crate_name}")?;
                        package_fmt(f, package, &format!("  {tab}"))?;
                    }
                }
            }

            Ok(())
        }

        writeln!(f, "Project(")?;
        writeln!(f, "  project_root: {:?}", self.project_root)?;
        writeln!(f, "  members:")?;
        for p in &self.nargo_workspace.members {
            package_fmt(f, p, "    ")?;
        }
        writeln!(f, "  loaded_files:")?;
        let file_map = self.nargo_file_manager.as_file_map();
        for file_id in file_map.all_file_ids().sorted() {
            writeln!(
                f,
                "    file_id: {:?}, name: {:?}",
                file_id,
                file_map.get_name(*file_id).unwrap()
            )?;
        }
        writeln!(f, ")")?;

        Ok(())
    }
}
