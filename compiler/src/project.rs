use std::{
    fmt::Debug,
    fs,
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
use noirc_frontend::{
    ast::{
        BlockExpression, CallExpression, Expression, ExpressionKind, FunctionKind, Ident,
        NoirFunction, Path as AstPath, PathKind, PathSegment, Pattern, Statement, StatementKind,
    },
    hir::ParsedFiles,
    parser::{ItemKind, ParsedModule},
    token::FunctionAttributeKind,
};

use crate::{collections::HashSet, error::Error};

#[derive(Clone, Copy)]
struct DependencySourceReplacement {
    trigger: &'static str,
    replacement: &'static str,
    manifest: &'static str,
    package: &'static str,
    source_overrides: &'static [DependencySourceOverride],
}

#[derive(Clone, Copy)]
struct DependencySourceOverride {
    dependency: &'static str,
    path: &'static str,
    expected: &'static str,
    replacement: &'static str,
}

// `noir-bignum` repeats a constrained relation check as an assertion inside its unconstrained
// quotient hint. Invalid ECDSA inputs in an inactive circuit branch can reach that hint before
// the branch guard masks the constrained checks, so the debugging assertion makes an otherwise
// total black-box operation panic. The relation remains enforced by the gadget's range checks.
// Match the complete pinned source line so a dependency update fails loudly instead of silently
// applying this compatibility override to changed code.
const ECDSA_DEPENDENCY_SOURCE_OVERRIDES: &[DependencySourceOverride] =
    &[DependencySourceOverride {
        dependency: "bignum",
        path: "src/fns/expressions.nr",
        expected: "    assert(__is_zero(remainder));\n",
        replacement: "",
    }];

/// Circuit-backed replacements that live in standalone Noir packages.
///
/// Unlike `FOREIGN_REPLACEMENTS`, these cannot be called from a rewritten function in the
/// embedded `std` crate: package dependencies are visible to the user's crate, not to `std`.
/// Keep that frontend boundary isolated here until low-level calls can target circuit functions
/// directly after name resolution.
const DEPENDENCY_SOURCE_REPLACEMENTS: &[DependencySourceReplacement] =
    &[DependencySourceReplacement {
        trigger: "std::ecdsa_secp256r1::verify_signature",
        replacement: "mavros_ecdsa::verify_signature",
        manifest: "../mavros_stdlib/ecdsa_dependencies/Nargo.toml",
        package: "mavros_ecdsa",
        source_overrides: ECDSA_DEPENDENCY_SOURCE_OVERRIDES,
    }];

pub struct Project {
    project_root: PathBuf,
    nargo_workspace: Workspace,
    nargo_file_manager: FileManager,
    nargo_parsed_files: ParsedFiles,
}

/// Mavros stdlib extensions that get injected into the `std/` namespace.
const MAVROS_STDLIB_FILES: &[(&str, &str)] = &[
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
        "std/mavros/replacements/embedded_curve_add.nr",
        include_str!("../../mavros_stdlib/replacements/embedded_curve_add.nr"),
    ),
    (
        "std/mavros/replacements/multi_scalar_mul.nr",
        include_str!("../../mavros_stdlib/replacements/multi_scalar_mul.nr"),
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
const FOREIGN_REPLACEMENTS: &[&str] = &[
    "blake3",
    "embedded_curve_add",
    "multi_scalar_mul",
    "poseidon2_permutation",
    "sha256_compression",
];

/// Rewrite all registered `#[foreign]` shims in the parsed files to call their replacements.
fn replace_foreign_functions(parsed_files: &mut ParsedFiles) {
    let mut replaced: HashSet<&'static str> = HashSet::default();
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

fn parse_workspace(
    workspace: &Workspace,
    source_replacements: &[DependencySourceReplacement],
) -> Result<(FileManager, ParsedFiles), Error> {
    // Build the file manager manually so we can expose the Mavros extensions from the embedded
    // stdlib root without maintaining a copy of upstream's `std/lib.nr`.
    let mut file_manager = FileManager::new(&workspace.root_dir);

    // 1. Add the embedded stdlib, extending its root with the Mavros module declaration.
    let stdlib_root = Path::new("std/lib.nr");
    let mut extended_stdlib_root = false;
    for (path, mut source) in stdlib_paths_with_source() {
        if Path::new(&path) == stdlib_root {
            source.push_str("\npub mod mavros;\n");
            extended_stdlib_root = true;
        }
        file_manager.add_file_with_source_canonical_path(Path::new(&path), source);
    }
    assert!(extended_stdlib_root, "embedded stdlib root was not found");

    // 2. Add the Mavros stdlib extensions.
    for (path, source) in MAVROS_STDLIB_FILES {
        file_manager.add_file_with_source_canonical_path(Path::new(path), source.to_string());
    }

    // 3. Rewrite calls backed by standalone circuit packages. Source files are inserted first so
    // nargo's subsequent bulk insertion preserves these overrides.
    for package in &workspace.members {
        add_dependency_rewritten_sources(
            &package.root_dir.join("src"),
            source_replacements,
            &mut file_manager,
        );
    }
    add_dependency_source_overrides(workspace, source_replacements, &mut file_manager)?;

    // 4. Add all remaining workspace and dependency files.
    nargo::insert_all_files_for_workspace_into_file_manager(workspace, &mut file_manager);
    let mut parsed_files = nargo::parse_all(&file_manager);

    // 5. Rewrite replaced foreign functions to call their pure-Noir implementations.
    replace_foreign_functions(&mut parsed_files);
    Ok((file_manager, parsed_files))
}

fn add_dependency_source_overrides(
    workspace: &Workspace,
    replacements: &[DependencySourceReplacement],
    file_manager: &mut FileManager,
) -> Result<(), Error> {
    for replacement in replacements {
        let replacement_package = workspace
            .members
            .iter()
            .find_map(|member| find_direct_dependency(member, replacement.package))
            .ok_or_else(|| {
                Error::DependencySourceOverride(format!(
                    "active source replacement package '{}' is not attached to the workspace",
                    replacement.package
                ))
            })?;

        for source_override in replacement.source_overrides {
            let dependency = find_dependency(replacement_package, source_override.dependency)
                .ok_or_else(|| {
                    Error::DependencySourceOverride(format!(
                        "replacement package '{}' has no '{}' dependency",
                        replacement.package, source_override.dependency
                    ))
                })?;
            let path = dependency.root_dir.join(source_override.path);
            let source = fs::read_to_string(&path).map_err(|error| {
                Error::DependencySourceOverride(format!(
                    "failed to read pinned source {}: {error}",
                    path.display()
                ))
            })?;
            if source.matches(source_override.expected).count() != 1 {
                return Err(Error::DependencySourceOverride(format!(
                    "pinned compatibility source changed in {}",
                    path.display()
                )));
            }
            let rewritten =
                source.replacen(source_override.expected, source_override.replacement, 1);
            file_manager.add_file_with_source_canonical_path(&path, rewritten);
        }
    }
    Ok(())
}

fn find_direct_dependency<'a>(package: &'a Package, name: &str) -> Option<&'a Package> {
    package
        .dependencies
        .iter()
        .find(|(dependency_name, _)| dependency_name.to_string() == name)
        .map(|(_, dependency)| dependency.package())
}

fn find_dependency<'a>(package: &'a Package, name: &str) -> Option<&'a Package> {
    if package.name.to_string() == name {
        return Some(package);
    }
    package
        .dependencies
        .values()
        .find_map(|dependency| find_dependency(dependency.package(), name))
}

fn add_dependency_rewritten_sources(
    dir: &Path,
    replacements: &[DependencySourceReplacement],
    file_manager: &mut FileManager,
) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            add_dependency_rewritten_sources(&path, replacements, file_manager);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("nr") {
            let Ok(source) = fs::read_to_string(&path) else {
                continue;
            };
            let rewritten = replacements
                .iter()
                .fold(source.clone(), |source, replacement| {
                    source.replace(replacement.trigger, replacement.replacement)
                });
            if rewritten != source {
                file_manager.add_file_with_source_canonical_path(&path, rewritten);
            }
        }
    }
}

fn source_tree_contains(dir: &Path, needle: &str) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if source_tree_contains(&path, needle) {
                return true;
            }
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("nr")
            && fs::read_to_string(&path).is_ok_and(|source| source.contains(needle))
        {
            return true;
        }
    }
    false
}

impl Project {
    pub fn new(project_root: PathBuf) -> Result<Self, Error> {
        // Workspace loading was done based on https://github.com/noir-lang/noir/blob/c3a43abf9be80c6f89560405b65f5241ed67a6b2/tooling/nargo_cli/src/cli/mod.rs#L180
        let toml_path = nargo_toml::get_package_manifest(&project_root)?;

        let mut nargo_workspace = nargo_toml::resolve_workspace_from_toml(&toml_path, All, None)?;

        // Nargo eagerly parses every declared dependency. Attach a replacement package only when
        // its trigger occurs in a workspace source tree; eagerly attaching a large circuit graph
        // penalizes unrelated compilations and can exhaust the small stack of Rust test threads.
        let source_replacements: Vec<_> = DEPENDENCY_SOURCE_REPLACEMENTS
            .iter()
            .copied()
            .filter(|replacement| {
                nargo_workspace.members.iter().any(|package| {
                    source_tree_contains(&package.root_dir.join("src"), replacement.trigger)
                })
            })
            .collect();
        for replacement in &source_replacements {
            let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join(replacement.manifest);
            let replacement_workspace =
                nargo_toml::resolve_workspace_from_toml(&manifest, All, None)?;
            let replacement_package = replacement_workspace.members[0].clone();
            for package in &mut nargo_workspace.members {
                package
                    .dependencies
                    .entry(replacement_package.name.clone())
                    .or_insert_with(|| Dependency::Local {
                        package: replacement_package.clone(),
                    });
            }
        }

        let (nargo_file_manager, nargo_parsed_files) =
            parse_workspace(&nargo_workspace, &source_replacements)?;

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

    /// Root directory of the package being compiled. For a workspace this is
    /// the member's directory, not the workspace root — `Prover.toml` and
    /// other per-package files live here (matching nargo's behaviour).
    pub fn package_root(&self) -> &Path {
        &self.get_only_crate().root_dir
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dependency_detection_matches_the_source_rewrite_trigger() {
        let project = tempfile::tempdir().unwrap();
        let src = project.path().join("src");
        let nested = src.join("nested");
        let replacement = DEPENDENCY_SOURCE_REPLACEMENTS[0];
        fs::create_dir_all(&nested).unwrap();

        fs::write(src.join("main.nr"), "fn main() {}\n").unwrap();
        fs::write(nested.join("ignored.txt"), replacement.trigger).unwrap();
        assert!(!source_tree_contains(&src, replacement.trigger));

        fs::write(
            nested.join("signature.nr"),
            format!("fn verify() {{ let _ = {}; }}\n", replacement.trigger),
        )
        .unwrap();
        assert!(source_tree_contains(&src, replacement.trigger));
    }

    #[test]
    fn project_without_ecdsa_does_not_attach_the_circuit_dependency() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("src")).unwrap();
        fs::write(
            root.path().join("Nargo.toml"),
            "[package]\nname = \"no_ecdsa\"\ntype = \"bin\"\nauthors = []\n\n[dependencies]\n",
        )
        .unwrap();
        fs::write(root.path().join("src/main.nr"), "fn main() {}\n").unwrap();

        let project = Project::new(root.path().to_path_buf()).unwrap();
        assert!(
            project
                .get_only_crate()
                .dependencies
                .keys()
                .all(|name| name.to_string() != "mavros_ecdsa")
        );
    }
}
