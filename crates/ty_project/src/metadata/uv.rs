use std::path::PathBuf;

use pep440_rs::VersionSpecifiers;
use ruff_db::system::{System, SystemPath, SystemPathBuf};
use ruff_ranged_value::{RangedValue, ValueSource};
use serde::Deserialize;
use thiserror::Error;
use ty_static::EnvVars;

use super::pyproject::{ResolveRequiresPythonError, resolve_requires_python_lower_bound};
use super::python_version::SupportedPythonVersion;

#[derive(Debug, Clone, PartialEq, Eq, get_size2::GetSize)]
pub struct UvWorkspace {
    root: SystemPathBuf,
    member: Option<SystemPathBuf>,
    environment: Option<SystemPathBuf>,
    requires_python: Option<RangedValue<SupportedPythonVersion>>,
}

impl UvWorkspace {
    pub fn discover(path: &SystemPath, system: &dyn System) -> Option<Self> {
        let uv = system
            .env_var(EnvVars::UV)
            .unwrap_or_else(|_| "uv".to_string());

        let output = match system.run_command(&uv, &["workspace", "metadata", "--sync"], path) {
            Ok(output) => output,
            Err(error) => {
                tracing::debug!("Failed to invoke `uv workspace metadata`: {error}");
                return None;
            }
        };

        if !output.status.success() {
            tracing::debug!(
                "`uv workspace metadata` failed with status {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
            return None;
        }

        match Self::from_metadata(path, &output.stdout, system) {
            Ok(workspace) => Some(workspace),
            Err(error) => {
                tracing::debug!("Failed to use `uv workspace metadata` output: {error}");
                None
            }
        }
    }

    pub fn from_metadata(
        path: &SystemPath,
        metadata: &[u8],
        system: &dyn System,
    ) -> Result<Self, UvWorkspaceError> {
        let metadata = serde_json::from_slice::<WorkspaceMetadata>(metadata)
            .map_err(UvWorkspaceError::InvalidMetadata)?;

        if metadata.schema.version != "preview" {
            return Err(UvWorkspaceError::UnsupportedSchemaVersion(
                metadata.schema.version,
            ));
        }

        let requires_python = resolve_requires_python_lower_bound(&RangedValue::new(
            metadata.requires_python,
            ValueSource::UvWorkspace,
        ))
        .map_err(UvWorkspaceError::InvalidRequiresPython)?;

        let root = existing_directory(metadata.workspace_root, "workspace root", system)?;
        if !path.starts_with(&root) {
            return Err(UvWorkspaceError::WorkspaceRootNotAncestor {
                root,
                path: path.to_path_buf(),
            });
        }

        let environment = metadata
            .environment
            .map(|environment| existing_directory(environment.root, "environment root", system))
            .transpose()?;

        // Workspace members may be nested. Select the deepest member containing `path`, which is
        // the package the user is currently operating in.
        let member = metadata
            .members
            .into_iter()
            .map(|member| member.path)
            .filter(|member| path.as_std_path().starts_with(member))
            .max_by_key(|member| member.as_os_str().len());
        let member = match member {
            Some(member) => Some(existing_directory(member, "workspace member", system)?),
            None => None,
        };

        Ok(Self {
            root,
            member,
            environment,
            requires_python,
        })
    }

    pub fn root(&self) -> &SystemPath {
        &self.root
    }

    pub fn member(&self) -> Option<&SystemPath> {
        self.member.as_deref()
    }

    pub fn environment(&self) -> Option<&SystemPath> {
        self.environment.as_deref()
    }

    pub fn requires_python(&self) -> Option<&RangedValue<SupportedPythonVersion>> {
        self.requires_python.as_ref()
    }
}

fn existing_directory(
    path: PathBuf,
    description: &'static str,
    system: &dyn System,
) -> Result<SystemPathBuf, UvWorkspaceError> {
    let path = match SystemPathBuf::from_path_buf(path) {
        Ok(path) => path,
        Err(path) => return Err(UvWorkspaceError::NonUnicodePath { description, path }),
    };

    if !system.is_directory(&path) {
        return Err(UvWorkspaceError::MissingDirectory { description, path });
    }

    Ok(path)
}

#[derive(Debug, Error)]
pub enum UvWorkspaceError {
    #[error("invalid `uv workspace metadata` JSON: {0}")]
    InvalidMetadata(serde_json::Error),

    #[error("unsupported `uv workspace metadata` schema version `{0}`")]
    UnsupportedSchemaVersion(String),

    #[error("invalid `requires_python` returned by `uv workspace metadata`: {0}")]
    InvalidRequiresPython(ResolveRequiresPythonError),

    #[error("non-Unicode {description} returned by `uv workspace metadata`: `{path}`", path = path.display())]
    NonUnicodePath {
        description: &'static str,
        path: PathBuf,
    },

    #[error("missing {description} returned by `uv workspace metadata`: `{path}`")]
    MissingDirectory {
        description: &'static str,
        path: SystemPathBuf,
    },

    #[error("uv workspace root `{root}` is not an ancestor of `{path}`")]
    WorkspaceRootNotAncestor {
        root: SystemPathBuf,
        path: SystemPathBuf,
    },
}

#[derive(Deserialize)]
struct WorkspaceMetadata {
    schema: WorkspaceMetadataSchema,
    workspace_root: PathBuf,
    environment: Option<WorkspaceEnvironment>,
    requires_python: VersionSpecifiers,
    #[serde(default)]
    members: Vec<WorkspaceMember>,
}

#[derive(Deserialize)]
struct WorkspaceMetadataSchema {
    version: String,
}

#[derive(Deserialize)]
struct WorkspaceEnvironment {
    root: PathBuf,
}

#[derive(Deserialize)]
struct WorkspaceMember {
    path: PathBuf,
}

#[cfg(test)]
mod tests {
    use ruff_db::system::{SystemPath, TestSystem};

    use super::{UvWorkspace, UvWorkspaceError};

    #[test]
    fn rejects_invalid_metadata() {
        let system = TestSystem::default();

        assert!(matches!(
            UvWorkspace::from_metadata(SystemPath::new("/app"), b"{", &system),
            Err(UvWorkspaceError::InvalidMetadata(_))
        ));
    }

    #[test]
    fn rejects_unsupported_schema() {
        let system = TestSystem::default();
        let metadata = br#"{
            "schema": { "version": "future" },
            "workspace_root": "/app",
            "requires_python": ">=3.8"
        }"#;

        assert!(matches!(
            UvWorkspace::from_metadata(SystemPath::new("/app"), metadata, &system),
            Err(UvWorkspaceError::UnsupportedSchemaVersion(version)) if version == "future"
        ));
    }

    #[test]
    fn members_can_be_omitted() -> anyhow::Result<()> {
        let system = TestSystem::default();
        system
            .memory_file_system()
            .write_file_all("/app/pyproject.toml", "[tool.uv.workspace]")?;
        let metadata = br#"{
            "schema": { "version": "preview" },
            "workspace_root": "/app",
            "requires_python": ">=3.8"
        }"#;

        let workspace = UvWorkspace::from_metadata(SystemPath::new("/app"), metadata, &system)?;

        assert!(workspace.member().is_none());

        Ok(())
    }
}
