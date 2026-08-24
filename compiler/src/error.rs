use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    NoirProjectError(#[from] nargo_toml::ManifestError),
    #[error("Package {package_name}:{} is missing dependency {dependency_name} in configuration", .package_version.as_ref().unwrap_or(&"missing".to_string()))]
    MissingDependencyError {
        package_name: String,
        package_version: Option<String>,
        dependency_name: String,
    },
    #[error("Package {package_name}:{} , dependency {dependency_name} - url parsing error {err}", .package_version.as_ref().unwrap_or(&"missing".to_string()))]
    URLParsingError {
        package_name: String,
        package_version: Option<String>,
        dependency_name: String,
        err: String,
    },
}
