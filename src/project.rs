use std::{fmt::Debug, path::{Path, PathBuf}};

use fm::FileManager;
use itertools::Itertools;
use nargo::{
    package::{Dependency, Package},
    workspace::Workspace,
};
use nargo_toml::PackageSelection::All;
use noirc_frontend::hir::ParsedFiles;

use crate::error::Error;

pub struct Project {
    project_root: PathBuf,
    nargo_workspace: Workspace,
    nargo_file_manager: FileManager,
    nargo_parsed_files: ParsedFiles,
}

pub struct ForeignFunctionImpl {
    pub file: &'static str,
    pub source: &'static str,
    pub custom_name: &'static str,
    pub stdlib_internal_name: &'static str,
}

pub const FOREIGN_FUNCTION_IMPLS: &[ForeignFunctionImpl] = &[
    ForeignFunctionImpl {
        file: "poseidon2_permutation.nr",
        source: include_str!("../stdlib_noir_impls/poseidon2_permutation.nr"),
        custom_name: "poseidon2_permutation",
        stdlib_internal_name: "poseidon2_permutation_internal",
    },
];

fn parse_workspace(workspace: &Workspace) -> (FileManager, ParsedFiles) {
    let mut file_manager = workspace.new_file_manager();
    nargo::insert_all_files_for_workspace_into_file_manager(workspace, &mut file_manager);
    for implementation in FOREIGN_FUNCTION_IMPLS {
        file_manager.add_file_with_source_canonical_path(Path::new(implementation.file), implementation.source.to_string());
    }

    let parsed_files = nargo::parse_all(&file_manager);
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
