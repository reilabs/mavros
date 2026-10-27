//! Witness metadata and the versioned `program.mavros` artifact.
//!
//! ```text
//! [ MAVR1CS\0 (8 bytes) ][ format version: u32 LE ][ compiler: u32 LE length, UTF-8 bytes ]
//! [ field: u32 LE ][ artifact digest: 32 bytes ][ bincode { manifest, r1cs, bytecode, abi } ]
//! ```
//!
//! The magic, format version and compiler version form the [`ArtifactPreamble`], laid out the same
//! in every format version so that a reader can name the compiler that wrote a file it cannot
//! otherwise open. Everything after it belongs to [`ARTIFACT_FORMAT_VERSION`]; bump that when
//! changing the header, the serialized body or the layouts.
//!
//! Two digests identify an artifact. [`Artifact::circuit_digest`] binds the field, the R1CS and the
//! witness manifest: the constraint system a proof is about. [`Artifact::digest`] binds the circuit
//! digest, the VM bytecode and the ABI: the whole file. The header carries the latter and reading
//! verifies it; the header itself is checked, not hashed, and the compiler version is informational.
//! Reading also checks the manifest and the bytecode header against the R1CS and refuses bytes after
//! the body.
//!
//! The manifest records only what the R1CS cannot say: each parameter's name, visibility, width
//! and witness columns, and the return's. Witness and constraint counts come from the R1CS
//! layouts, and LogUp challenge `i` sits at `witness_layout.challenges_start() + i`, which is
//! offset `i` into the post-commitment witness.
//!
//! The entry blob holds the parameters only, in ABI order. Witness columns follow the wrapper's
//! write order instead: column 0 is the constant one, then the public parameters in ABI order,
//! then the private parameters that survived dead-code elimination in ABI order, and the computed
//! return after `main`'s internal columns.

use std::ops::Range;

use bincode::Options as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::{
    Field, FieldId, MavrosField, R1CS, SealedR1CS, field_tag,
    program_header::{ENTRY_AD, ENTRY_WITGEN, NO_ENTRY_POINT, ProgramHeader},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ParamVisibility {
    Public,
    Private,
    /// Noir's `call_data` / `return_data`, handled exactly like `Private`.
    DataBus,
}

impl ParamVisibility {
    fn is_public(self) -> bool {
        matches!(self, ParamVisibility::Public)
    }
}

/// The entry blob slot is the sum of the widths before it: [`WitnessManifest::blob_slots`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestParam {
    pub name: String,
    pub visibility: ParamVisibility,
    /// Flattened field count.
    pub width: usize,
    /// `None` for an elided private or zero-width parameter.
    pub column_start: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReturnSlot {
    /// `None` for a zero-width return.
    pub column_start: Option<usize>,
    pub len: usize,
    pub visibility: ParamVisibility,
}

/// Where `main`'s parameters and return sit in the entry blob and the witness.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WitnessManifest {
    /// In ABI order.
    pub parameters: Vec<ManifestParam>,
    /// Present exactly when the ABI declares a return.
    pub returns: Option<ReturnSlot>,
}

impl WitnessManifest {
    /// Flattened width of the entry blob: the parameters in ABI order.
    pub fn entry_blob_field_count(&self) -> usize {
        self.parameters.iter().map(|param| param.width).sum()
    }

    /// Each parameter with the entry blob slots it occupies, in ABI order.
    pub fn blob_slots(&self) -> impl Iterator<Item = (&ManifestParam, Range<usize>)> {
        let mut offset = 0;
        self.parameters.iter().map(move |param| {
            let start = offset;
            offset += param.width;
            (param, start..offset)
        })
    }

    /// Checks the parameter and return columns against `r1cs`'s witness layout.
    pub fn validate<F: MavrosField>(&self, r1cs: &R1CS<F>) -> Result<(), String> {
        let layout = r1cs.witness_layout;
        let check = |ok: bool, what: &str| {
            ok.then_some(())
                .ok_or_else(|| format!("witness manifest disagrees with the R1CS: {what}"))
        };
        let add = |start: usize, len: usize, what: &str| {
            start.checked_add(len).ok_or_else(|| {
                format!("witness manifest disagrees with the R1CS: {what} overflows")
            })
        };
        self.parameters
            .iter()
            .try_fold(0, |total, param| add(total, param.width, "entry blob"))?;

        // Columns after the constant one: public parameters, then the surviving private ones.
        let mut column = 1;
        for public in [true, false] {
            for param in &self.parameters {
                if param.visibility.is_public() != public {
                    continue;
                }
                if param.width == 0 {
                    check(param.column_start.is_none(), "zero-width parameter column")?;
                    continue;
                }
                match param.column_start {
                    None => check(!public, "public parameter without columns")?,
                    Some(start) => {
                        check(start == column, "parameter column")?;
                        column = add(column, param.width, "parameter columns")?;
                    }
                }
            }
        }
        check(
            column <= layout.algebraic_size,
            "parameter columns outside the algebraic section",
        )?;
        if let Some(ret) = self.returns {
            check(
                (ret.len == 0) == ret.column_start.is_none(),
                "zero-width return column",
            )?;
            if let Some(start) = ret.column_start {
                check(start >= column, "return column")?;
                check(
                    add(start, ret.len, "return columns")? <= layout.algebraic_size,
                    "return columns outside the algebraic section",
                )?;
            }
        }
        Ok(())
    }
}

pub const ARTIFACT_MAGIC: [u8; 8] = *b"MAVR1CS\0";
/// Format version this crate writes and the only one it reads.
pub const ARTIFACT_FORMAT_VERSION: u32 = 2;
/// The first format whose header records the compiler version; format 1 recorded none.
const COMPILER_SINCE_VERSION: u32 = 2;

/// Domain separator of [`Artifact::circuit_digest`]. Its bytes and preimage are those of the
/// `r1cs.bin` digest this format replaced, so a circuit keeps the digest it had there. Bump when
/// the preimage encoding changes.
pub const CIRCUIT_DIGEST_DOMAIN: &[u8] = b"mavros-r1cs-artifact-v1\0";

/// Domain separator of [`Artifact::digest`]. Bump when its preimage encoding changes.
pub const ARTIFACT_DIGEST_DOMAIN: &[u8] = b"mavros-artifact-v1\0";

const VERSION_OFFSET: usize = ARTIFACT_MAGIC.len();
const COMPILER_LEN_OFFSET: usize = VERSION_OFFSET + 4;
const COMPILER_OFFSET: usize = COMPILER_LEN_OFFSET + 4;
const FIELD_LEN: usize = 4;
const DIGEST_LEN: usize = 32;

#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error(
        "the file does not start with a well-formed MAVR1CS header, so it is not a Mavros artifact"
    )]
    MissingHeader,
    #[error(
        "unsupported artifact format version {found}, written by {}; this reader expects \
         {ARTIFACT_FORMAT_VERSION}",
        writer(.compiler)
    )]
    UnsupportedVersion {
        found: u32,
        /// `None` for a format that recorded no compiler version.
        compiler: Option<String>,
    },
    #[error("the artifact names unknown field {0}")]
    UnknownField(u32),
    #[error("the artifact is over {found:?} but {expected:?} was requested")]
    WrongField { expected: FieldId, found: FieldId },
    #[error("artifact body: {0}")]
    Body(#[from] bincode::Error),
    #[error("artifact digest mismatch: the header says {expected} but the body hashes to {actual}")]
    DigestMismatch { expected: String, actual: String },
    #[error("{0}")]
    Invalid(String),
    #[error("artifact bytecode: {0}")]
    Bytecode(String),
}

fn body_options() -> impl bincode::Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
}

/// Lowercase hex, the form digests are printed and recorded in.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn writer(compiler: &Option<String>) -> String {
    compiler.as_deref().map_or_else(
        || "an unknown compiler".to_string(),
        |compiler| format!("mavros {compiler}"),
    )
}

fn u32_at(bytes: &[u8], start: usize) -> Result<u32, ArtifactError> {
    bytes
        .get(start..start + 4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("4 bytes")))
        .ok_or(ArtifactError::MissingHeader)
}

/// The start of every artifact, laid out the same in every format version: enough to name the
/// compiler that wrote the file before, or instead of, decoding the rest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactPreamble {
    pub format_version: u32,
    /// The release version of the compiler that wrote the artifact; `None` for a format that
    /// recorded none.
    pub compiler: Option<String>,
}

impl ArtifactPreamble {
    /// Needs only the magic, so it reads an artifact of any format version.
    pub fn peek(bytes: &[u8]) -> Result<Self, ArtifactError> {
        if bytes.get(..ARTIFACT_MAGIC.len()) != Some(&ARTIFACT_MAGIC[..]) {
            return Err(ArtifactError::MissingHeader);
        }
        let format_version = u32_at(bytes, VERSION_OFFSET)?;
        let compiler = if format_version >= COMPILER_SINCE_VERSION {
            let len = u32_at(bytes, COMPILER_LEN_OFFSET)? as usize;
            let compiler = COMPILER_OFFSET
                .checked_add(len)
                .and_then(|end| bytes.get(COMPILER_OFFSET..end))
                .and_then(|compiler| std::str::from_utf8(compiler).ok())
                .ok_or(ArtifactError::MissingHeader)?;
            Some(compiler.to_string())
        } else {
            None
        };
        Ok(Self {
            format_version,
            compiler,
        })
    }

    /// Where the format-specific part of the header begins.
    pub(crate) fn encoded_len(&self) -> usize {
        match &self.compiler {
            Some(compiler) => COMPILER_OFFSET + compiler.len(),
            None => COMPILER_LEN_OFFSET,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(bound = "F: MavrosField")]
struct Body<F: MavrosField> {
    manifest: WitnessManifest,
    r1cs: R1CS<F>,
    bytecode: Vec<u64>,
    abi: Vec<u8>,
}

#[derive(Serialize)]
#[serde(bound = "F: MavrosField")]
struct BodyRef<'a, F: MavrosField> {
    manifest: &'a WitnessManifest,
    r1cs: &'a R1CS<F>,
    bytecode: &'a [u64],
    abi: &'a [u8],
}

#[derive(Debug)]
pub struct Artifact<F: MavrosField = Field> {
    compiler: String,
    manifest: WitnessManifest,
    r1cs: SealedR1CS<F>,
    bytecode: Vec<u64>,
    abi: Vec<u8>,
    circuit_digest: [u8; 32],
    digest: [u8; 32],
}

/// The owned contents of an [`Artifact`]; see [`Artifact::into_parts`].
#[derive(Debug)]
pub struct ArtifactParts<F: MavrosField = Field> {
    pub compiler: String,
    pub manifest: WitnessManifest,
    pub r1cs: SealedR1CS<F>,
    pub bytecode: Vec<u64>,
    pub abi: Vec<u8>,
}

impl<F: MavrosField> Artifact<F> {
    pub fn new(
        compiler: String,
        manifest: WitnessManifest,
        r1cs: SealedR1CS<F>,
        bytecode: Vec<u64>,
        abi: Vec<u8>,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(CIRCUIT_DIGEST_DOMAIN);
        hasher.update(r1cs.digest());
        bincode::serialize_into(&mut hasher, &manifest)
            .expect("hashing cannot fail: the hasher accepts every byte");
        let circuit_digest: [u8; 32] = hasher.finalize().into();
        let mut hasher = Sha256::new();
        hasher.update(ARTIFACT_DIGEST_DOMAIN);
        hasher.update(circuit_digest);
        bincode::serialize_into(&mut hasher, &(&bytecode, &abi))
            .expect("hashing cannot fail: the hasher accepts every byte");
        Self {
            compiler,
            manifest,
            r1cs,
            bytecode,
            abi,
            circuit_digest,
            digest: hasher.finalize().into(),
        }
    }

    pub fn field(&self) -> FieldId {
        self.r1cs.field()
    }

    /// SHA-256 over [`ARTIFACT_DIGEST_DOMAIN`], [`Self::circuit_digest`] and the bincode-encoded
    /// bytecode and ABI: the identity of the whole file, which the header carries. Pin it to mean
    /// "these exact bytes".
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// SHA-256 over [`CIRCUIT_DIGEST_DOMAIN`], the field-bound [`R1CS::digest`] and the
    /// bincode-encoded manifest: the identity of the constraint system and its witness layout,
    /// which a codegen-only change leaves alone. Pin it to mean "the same circuit", as a verifier
    /// or a proof transcript should.
    pub fn circuit_digest(&self) -> [u8; 32] {
        self.circuit_digest
    }

    pub fn manifest(&self) -> &WitnessManifest {
        &self.manifest
    }

    pub fn r1cs(&self) -> &SealedR1CS<F> {
        &self.r1cs
    }

    pub fn bytecode(&self) -> &[u64] {
        &self.bytecode
    }

    pub fn abi(&self) -> &[u8] {
        &self.abi
    }

    pub fn compiler(&self) -> &str {
        &self.compiler
    }

    /// For a consumer that must own the R1CS without cloning it. Read [`Self::digest`] or
    /// [`Self::circuit_digest`] first; they are not among the parts.
    pub fn into_parts(self) -> ArtifactParts<F> {
        ArtifactParts {
            compiler: self.compiler,
            manifest: self.manifest,
            r1cs: self.r1cs,
            bytecode: self.bytecode,
            abi: self.abi,
        }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, ArtifactError> {
        let compiler_len = u32::try_from(self.compiler.len()).map_err(|_| {
            ArtifactError::Invalid("the compiler version does not fit in u32 bytes".to_string())
        })?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&ARTIFACT_MAGIC);
        bytes.extend_from_slice(&ARTIFACT_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&compiler_len.to_le_bytes());
        bytes.extend_from_slice(self.compiler.as_bytes());
        bytes.extend_from_slice(&field_tag(self.field()).to_le_bytes());
        bytes.extend_from_slice(&self.digest);
        body_options().serialize_into(
            &mut bytes,
            &BodyRef {
                manifest: &self.manifest,
                r1cs: self.r1cs.system(),
                bytecode: &self.bytecode,
                abi: &self.abi,
            },
        )?;
        Ok(bytes)
    }

    /// Reads an artifact, verifying its digest, then the manifest and the bytecode header against
    /// the R1CS. Trailing bytes are an error.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ArtifactError> {
        let preamble = ArtifactPreamble::peek(bytes)?;
        if preamble.format_version != ARTIFACT_FORMAT_VERSION {
            return Err(ArtifactError::UnsupportedVersion {
                found: preamble.format_version,
                compiler: preamble.compiler,
            });
        }
        let field_offset = preamble.encoded_len();
        let digest_offset = field_offset + FIELD_LEN;
        let body_offset = digest_offset + DIGEST_LEN;
        let field_code = u32_at(bytes, field_offset)?;
        let field = FieldId::from_code(u64::from(field_code))
            .ok_or(ArtifactError::UnknownField(field_code))?;
        if field != F::ID {
            return Err(ArtifactError::WrongField {
                expected: F::ID,
                found: field,
            });
        }
        let digest: [u8; 32] = bytes
            .get(digest_offset..body_offset)
            .ok_or(ArtifactError::MissingHeader)?
            .try_into()
            .expect("32 bytes");
        let body: Body<F> = body_options().deserialize(&bytes[body_offset..])?;
        let compiler = preamble
            .compiler
            .expect("the current format records the compiler version");
        let artifact = Self::new(
            compiler,
            body.manifest,
            SealedR1CS::new(body.r1cs),
            body.bytecode,
            body.abi,
        );
        if artifact.digest != digest {
            return Err(ArtifactError::DigestMismatch {
                expected: hex(&digest),
                actual: hex(&artifact.digest),
            });
        }
        artifact
            .manifest
            .validate(&artifact.r1cs)
            .map_err(ArtifactError::Invalid)?;
        check_bytecode::<F>(&artifact.bytecode, &artifact.r1cs, &artifact.manifest)
            .map_err(ArtifactError::Bytecode)?;
        Ok(artifact)
    }
}

/// The bytecode header must describe the program the R1CS and manifest describe.
fn check_bytecode<F: MavrosField>(
    bytecode: &[u64],
    r1cs: &R1CS<F>,
    manifest: &WitnessManifest,
) -> Result<(), String> {
    let header = ProgramHeader::parse(bytecode)?;
    if header.field != F::ID {
        return Err(format!(
            "the bytecode is over {:?} but the artifact is over {:?}",
            header.field,
            F::ID
        ));
    }
    if !header.has_layouts(r1cs.witness_layout, r1cs.constraints_layout) {
        return Err("the bytecode's constraint layouts disagree with the R1CS".to_string());
    }
    let len = bytecode.len();
    if header.code_start > len
        || header
            .entry_points
            .iter()
            .any(|&entry| entry != NO_ENTRY_POINT && entry >= len)
    {
        return Err("the bytecode's header points past its end".to_string());
    }
    for (entry, name) in [(ENTRY_WITGEN, "witgen"), (ENTRY_AD, "AD")] {
        if header
            .entry_points
            .get(entry)
            .is_none_or(|offset| *offset == NO_ENTRY_POINT)
        {
            return Err(format!(
                "the bytecode has no {name} entry point; a verifier binary is not a prover artifact"
            ));
        }
    }
    let blob = manifest.entry_blob_field_count();
    if header.entry_blob_field_count != blob {
        return Err(format!(
            "the bytecode reads an entry blob of {} fields but the manifest has {blob}",
            header.entry_blob_field_count
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ConstraintsLayout, Field, R1C, WitnessLayout,
        program_header::{ConstraintLayouts, PROGRAM_FORMAT_VERSION, PROGRAM_MAGIC},
    };

    /// `fn main(x: Field, y: pub Field) -> pub Field { x * y }`: columns are
    /// `[1, y, x, t3, t4, ret]`, with one constraint `y * x = ret`.
    fn r1cs() -> R1CS {
        R1CS {
            witness_layout: WitnessLayout {
                algebraic_size: 6,
                multiplicities_size: 0,
                challenges_size: 0,
                tables_data_size: 0,
                lookups_data_size: 0,
            },
            constraints_layout: ConstraintsLayout {
                algebraic_size: 1,
                tables_data_size: 0,
                lookups_data_size: 0,
            },
            constraints: vec![R1C {
                a: vec![(1, Field::from(1u64))],
                b: vec![(2, Field::from(1u64))],
                c: vec![(5, Field::from(1u64))],
            }],
        }
    }

    fn manifest() -> WitnessManifest {
        WitnessManifest {
            parameters: vec![
                ManifestParam {
                    name: "x".into(),
                    visibility: ParamVisibility::Private,
                    width: 1,
                    column_start: Some(2),
                },
                ManifestParam {
                    name: "y".into(),
                    visibility: ParamVisibility::Public,
                    width: 1,
                    column_start: Some(1),
                },
            ],
            returns: Some(ReturnSlot {
                column_start: Some(5),
                len: 1,
                visibility: ParamVisibility::Public,
            }),
        }
    }

    /// A header agreeing with `manifest` and `r1cs`: no constant pool or globals, witgen and AD
    /// entries at the one function marker that follows, no code.
    fn bytecode(manifest: &WitnessManifest, r1cs: &R1CS) -> Vec<u64> {
        let mut words = vec![
            PROGRAM_MAGIC,
            PROGRAM_FORMAT_VERSION,
            FieldId::Bn254.code(),
            1,
        ];
        words.extend(ConstraintLayouts::new(r1cs.witness_layout, r1cs.constraints_layout).words());
        let marker = words.len() as u64 + 6;
        words.extend([
            0,
            0,
            manifest.entry_blob_field_count() as u64,
            2,
            marker,
            marker,
        ]);
        words.push(u64::MAX);
        words
    }

    fn abi() -> Vec<u8> {
        b"{}".to_vec()
    }

    fn build(manifest: WitnessManifest, r1cs: R1CS, bytecode: Vec<u64>, abi: Vec<u8>) -> Artifact {
        Artifact::new(
            "test".into(),
            manifest,
            SealedR1CS::new(r1cs),
            bytecode,
            abi,
        )
    }

    fn artifact(manifest: WitnessManifest, r1cs: R1CS) -> Artifact {
        let bytecode = bytecode(&manifest, &r1cs);
        build(manifest, r1cs, bytecode, abi())
    }

    fn bytes_of(manifest: WitnessManifest, r1cs: R1CS) -> Vec<u8> {
        artifact(manifest, r1cs).to_bytes().unwrap()
    }

    fn read(bytes: &[u8]) -> Result<Artifact, ArtifactError> {
        Artifact::from_bytes(bytes)
    }

    fn round_trip(manifest: WitnessManifest, r1cs: R1CS) -> Result<Artifact, ArtifactError> {
        read(&bytes_of(manifest, r1cs))
    }

    fn digest_range(bytes: &[u8]) -> Range<usize> {
        let start = ArtifactPreamble::peek(bytes).unwrap().encoded_len() + FIELD_LEN;
        start..start + DIGEST_LEN
    }

    #[test]
    fn artifact_round_trips() {
        let cases: [fn(&mut WitnessManifest, &mut R1CS); 5] = [
            |_, _| {},
            |m, _| m.returns = None,
            // A zero-width return has no columns.
            |m, _| {
                let ret = m.returns.as_mut().unwrap();
                ret.len = 0;
                ret.column_start = None;
            },
            // An unused private parameter has no columns.
            |m, _| m.parameters[0].column_start = None,
            // A zero-width parameter has none either.
            |m, _| {
                m.parameters[0].width = 0;
                m.parameters[0].column_start = None;
            },
        ];
        for mutate in cases {
            let (mut m, mut r) = (manifest(), r1cs());
            mutate(&mut m, &mut r);
            let bytes = bytes_of(m.clone(), r.clone());
            assert!(bytes.starts_with(&ARTIFACT_MAGIC));
            assert_eq!(
                bytes,
                bytes_of(m.clone(), r.clone()),
                "serialization is deterministic"
            );
            let decoded = read(&bytes).unwrap();
            assert_eq!(decoded.field(), FieldId::Bn254);
            assert_eq!(decoded.r1cs().digest(), r.digest());
            assert_eq!(&bytes[digest_range(&bytes)], &decoded.digest());
            assert_ne!(decoded.digest(), decoded.circuit_digest());
            assert_ne!(decoded.circuit_digest(), decoded.r1cs().digest());
            assert_eq!(decoded.manifest(), &m);
            assert_eq!(decoded.bytecode(), bytecode(&m, &r));
            assert_eq!(decoded.abi(), abi());
            assert_eq!(decoded.compiler(), "test");
            assert_eq!(
                decoded.r1cs().witness_layout.size(),
                r.witness_layout.size()
            );
            assert_eq!(decoded.r1cs().constraints.len(), r.constraints.len());
            let parts = decoded.into_parts();
            assert_eq!(parts.manifest, m);
            assert_eq!(parts.compiler, "test");
        }
    }

    #[test]
    fn artifact_rejects_invalid_headers() {
        assert!(matches!(read(&[]), Err(ArtifactError::MissingHeader)));
        let mut bytes = bytes_of(manifest(), r1cs());
        let body = digest_range(&bytes).end;
        for cut in [COMPILER_OFFSET + 1, body - 1] {
            assert!(
                matches!(read(&bytes[..cut]), Err(ArtifactError::MissingHeader)),
                "cut at {cut}"
            );
        }
        for version in [0, ARTIFACT_FORMAT_VERSION - 1, ARTIFACT_FORMAT_VERSION + 1] {
            bytes[VERSION_OFFSET..COMPILER_LEN_OFFSET].copy_from_slice(&version.to_le_bytes());
            let compiler = (version >= COMPILER_SINCE_VERSION).then(|| "test".to_string());
            assert!(matches!(
                read(&bytes),
                Err(ArtifactError::UnsupportedVersion { found, compiler: found_compiler })
                    if found == version && found_compiler == compiler
            ));
            assert_eq!(
                ArtifactPreamble::peek(&bytes).unwrap(),
                ArtifactPreamble {
                    format_version: version,
                    compiler,
                }
            );
        }
        bytes[VERSION_OFFSET..COMPILER_LEN_OFFSET]
            .copy_from_slice(&ARTIFACT_FORMAT_VERSION.to_le_bytes());
        let field = digest_range(&bytes).start - FIELD_LEN;
        bytes[field..field + FIELD_LEN].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(read(&bytes), Err(ArtifactError::UnknownField(99))));
        bytes[COMPILER_OFFSET] = 0xff;
        assert!(matches!(read(&bytes), Err(ArtifactError::MissingHeader)));

        // A format-1 file: the field tag and digest follow the version, and no compiler.
        let mut format_1 = ARTIFACT_MAGIC.to_vec();
        format_1.extend_from_slice(&1u32.to_le_bytes());
        format_1.extend_from_slice(&1u32.to_le_bytes());
        format_1.extend_from_slice(&[0xff; 32]);
        assert!(matches!(
            read(&format_1),
            Err(ArtifactError::UnsupportedVersion {
                found: 1,
                compiler: None,
            })
        ));
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut padded = bytes_of(manifest(), r1cs());
        padded.push(0);
        assert!(matches!(read(&padded), Err(ArtifactError::Body(_))));
    }

    #[test]
    fn artifact_rejects_inconsistent_entry_slots() {
        let cases: [fn(&mut WitnessManifest, &mut R1CS); 8] = [
            // A public parameter always has columns, and they come first.
            |m, _| m.parameters[1].column_start = None,
            |m, _| m.parameters[1].column_start = Some(2),
            |m, _| m.parameters[0].column_start = Some(1),
            |m, _| m.parameters[0].column_start = Some(3),
            // The return follows the parameters and stays inside the algebraic section.
            |m, _| m.returns.as_mut().unwrap().column_start = None,
            |m, _| m.returns.as_mut().unwrap().column_start = Some(2),
            |m, _| m.returns.as_mut().unwrap().len = 2,
            // Parameter columns past the algebraic section.
            |_, r| r.witness_layout.algebraic_size = 2,
        ];
        for (case, mutate) in cases.into_iter().enumerate() {
            let (mut m, mut r) = (manifest(), r1cs());
            mutate(&mut m, &mut r);
            assert!(
                matches!(round_trip(m, r), Err(ArtifactError::Invalid(_))),
                "entry slot case {case}"
            );
        }
    }

    #[test]
    fn artifact_rejects_overflowing_ranges_and_totals() {
        let cases: [fn(&mut WitnessManifest, &mut R1CS); 2] = [
            |m, _| m.parameters[0].width = usize::MAX,
            |m, _| m.returns.as_mut().unwrap().len = usize::MAX,
        ];
        for (case, mutate) in cases.into_iter().enumerate() {
            let (mut m, mut r) = (manifest(), r1cs());
            mutate(&mut m, &mut r);
            let bytes = build(m, r, bytecode(&manifest(), &r1cs()), abi())
                .to_bytes()
                .unwrap();
            assert!(
                matches!(read(&bytes), Err(ArtifactError::Invalid(_))),
                "overflow case {case}"
            );
        }
    }

    /// Words: 0 magic, 1 version, 2 field, 3 layout tag, 4..12 layouts, 12 pool length,
    /// 13 globals, 14 entry blob, 15 entry count, 16 witgen entry, 17 AD entry, 18 marker.
    #[test]
    fn artifact_rejects_bytecode_that_disagrees_with_the_circuit() {
        let cases: [(&str, fn(&mut Vec<u64>)); 10] = [
            ("empty", |b| b.clear()),
            ("no magic", |b| b[0] = 0),
            ("another format version", |b| b[1] += 1),
            ("another field", |b| b[2] = FieldId::Goldilocks.code()),
            ("another algebraic size", |b| b[4] += 1),
            ("another entry blob", |b| b[14] += 1),
            ("entry past the end", |b| b[16] = 99),
            ("no code after the header", |b| b.truncate(18)),
            ("no AD entry", |b| b[17] = NO_ENTRY_POINT as u64),
            ("witgen entry only", |b| {
                b[15] = 1;
                b.remove(17);
            }),
        ];
        for (case, mutate) in cases {
            let mut words = bytecode(&manifest(), &r1cs());
            mutate(&mut words);
            let bytes = build(manifest(), r1cs(), words, abi()).to_bytes().unwrap();
            assert!(
                matches!(read(&bytes), Err(ArtifactError::Bytecode(_))),
                "{case}"
            );
        }
    }

    /// Both digests are compatibility promises to consumers that record them: pin their encoding.
    #[test]
    fn digest_encoding_is_pinned() {
        assert_eq!(
            hex(&r1cs().digest()),
            "a1826c2cf819ba7805777209988b33f98071393e14c2671b9ddb3fe9fafe2c52"
        );
        let pinned = artifact(manifest(), r1cs());
        // The digest the r1cs.bin artifact carried for this circuit.
        assert_eq!(
            hex(&pinned.circuit_digest()),
            "91938a47b4fbf034bad07ba83f8289ca1703176a63fae4867f05c0cd4505de18"
        );
        assert_eq!(
            hex(&pinned.digest()),
            "9c577fef5ebe660bad1ad1974cc92d98c4655bcafaad820615679b897ac90b6c"
        );

        // The compiler version stays outside both digests.
        let other = Artifact::new(
            "other".into(),
            manifest(),
            SealedR1CS::new(r1cs()),
            bytecode(&manifest(), &r1cs()),
            abi(),
        );
        assert_eq!(other.circuit_digest(), pinned.circuit_digest());
        assert_eq!(other.digest(), pinned.digest());
        assert_eq!(
            read(&other.to_bytes().unwrap()).unwrap().compiler(),
            "other"
        );

        // Bytecode and ABI move the artifact digest only; the circuit moves both.
        let mut words = bytecode(&manifest(), &r1cs());
        *words.last_mut().unwrap() = 7;
        for other in [
            build(manifest(), r1cs(), words, abi()),
            build(
                manifest(),
                r1cs(),
                bytecode(&manifest(), &r1cs()),
                b"{ }".to_vec(),
            ),
        ] {
            assert_eq!(other.circuit_digest(), pinned.circuit_digest());
            assert_ne!(other.digest(), pinned.digest());
        }
        let mut renamed = manifest();
        renamed.parameters[0].name = "another_input".into();
        let other = artifact(renamed, r1cs());
        assert_ne!(other.circuit_digest(), pinned.circuit_digest());
        assert_ne!(other.digest(), pinned.digest());
    }

    /// A flipped header byte, another R1CS, bytecode or ABI under the honest digest, or a
    /// manifest edit that still validates: each is a digest mismatch.
    #[test]
    fn tampering_is_rejected() {
        let honest = bytes_of(manifest(), r1cs());
        let rejects = |bytes: &[u8], case: &str| {
            assert!(
                matches!(read(bytes), Err(ArtifactError::DigestMismatch { .. })),
                "{case}"
            );
        };
        let with_honest_digest = |mut forged: Vec<u8>| {
            let range = digest_range(&honest);
            forged[range.clone()].copy_from_slice(&honest[range]);
            forged
        };

        let mut flipped = honest.clone();
        flipped[digest_range(&honest).start] ^= 1;
        rejects(&flipped, "flipped digest byte");

        let mut other = r1cs();
        other.constraints[0].c[0].1 = Field::from(2u64);
        rejects(
            &with_honest_digest(bytes_of(manifest(), other)),
            "changed coefficient",
        );

        let with_parts = |bytecode: Vec<u64>, abi: Vec<u8>| {
            build(manifest(), r1cs(), bytecode, abi).to_bytes().unwrap()
        };
        let mut words = bytecode(&manifest(), &r1cs());
        *words.last_mut().unwrap() = 7;
        rejects(
            &with_honest_digest(with_parts(words, abi())),
            "changed bytecode word",
        );
        rejects(
            &with_honest_digest(with_parts(bytecode(&manifest(), &r1cs()), b"{ }".to_vec())),
            "changed ABI byte",
        );

        let variants: [fn(&mut WitnessManifest); 6] = [
            |m| m.returns.as_mut().unwrap().column_start = Some(3),
            |m| m.returns.as_mut().unwrap().visibility = ParamVisibility::Private,
            |m| m.returns = None,
            |m| m.parameters[0].column_start = None,
            |m| m.parameters[0].name = "another_input".into(),
            |m| {
                m.parameters[0].visibility = ParamVisibility::Public;
                m.parameters[0].column_start = Some(1);
                m.parameters[1].visibility = ParamVisibility::Private;
                m.parameters[1].column_start = Some(2);
            },
        ];
        for (case, mutate) in variants.into_iter().enumerate() {
            let mut changed = manifest();
            mutate(&mut changed);
            changed.validate(&r1cs()).unwrap();
            rejects(
                &with_honest_digest(bytes_of(changed, r1cs())),
                &format!("manifest change {case}"),
            );
        }
    }

    /// A coefficient outside the field is a decoding error, never a panic.
    #[test]
    fn non_canonical_coefficient_is_a_body_error() {
        let honest = bytes_of(manifest(), r1cs());
        let one = Field::from(1u64);
        let limbs = bincode::serialize(&(5usize, [u64::MAX; 4])).unwrap();
        let honest_limbs = bincode::serialize(&(5usize, {
            use ark_ff::PrimeField as _;
            one.into_bigint().0
        }))
        .unwrap();
        let at = honest
            .windows(honest_limbs.len())
            .position(|window| window == honest_limbs.as_slice())
            .expect("the c term is encoded in the body");
        let mut bytes = honest.clone();
        bytes[at..at + limbs.len()].copy_from_slice(&limbs);
        assert!(matches!(read(&bytes), Err(ArtifactError::Body(_))));
    }
}
