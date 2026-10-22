//! Witness metadata and the versioned `program.mavros` artifact.
//!
//! ```text
//! [ MAVR1CS\0 (8 bytes) ][ format version: u32 LE ][ field: u32 LE ][ artifact digest: 32 bytes ]
//! [ bincode { compiler, manifest, r1cs, bytecode, abi } ]
//! ```
//!
//! The digest binds the field, R1CS, witness manifest, VM bytecode and ABI; reading verifies it
//! and refuses bytes after the body. The compiler version is informational and stays outside the
//! digest. Bump [`ARTIFACT_FORMAT_VERSION`] when changing the header, the serialized body or the
//! layouts.
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

use crate::{Field, FieldId, MavrosField, R1CS, SealedR1CS, field_tag};

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

/// Domain separator of [`Artifact::digest`]. Bump when its preimage encoding changes.
pub const ARTIFACT_DIGEST_DOMAIN: &[u8] = b"mavros-artifact-v1\0";

const VERSION_OFFSET: usize = ARTIFACT_MAGIC.len();
const FIELD_OFFSET: usize = VERSION_OFFSET + 4;
const DIGEST_OFFSET: usize = FIELD_OFFSET + 4;
const HEADER_LEN: usize = DIGEST_OFFSET + 32;

#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error("the file does not start with the MAVR1CS header, so it is not a Mavros artifact")]
    MissingHeader,
    #[error("unsupported artifact format version {0}; expected {ARTIFACT_FORMAT_VERSION}")]
    UnsupportedVersion(u32),
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
}

fn body_options() -> impl bincode::Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Debug, Deserialize)]
#[serde(bound = "F: MavrosField")]
struct Body<F: MavrosField> {
    compiler: String,
    manifest: WitnessManifest,
    r1cs: R1CS<F>,
    bytecode: Vec<u64>,
    abi: Vec<u8>,
}

#[derive(Serialize)]
#[serde(bound = "F: MavrosField")]
struct BodyRef<'a, F: MavrosField> {
    compiler: &'a str,
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
    digest: [u8; 32],
}

impl<F: MavrosField> Artifact<F> {
    pub fn new(
        manifest: WitnessManifest,
        r1cs: SealedR1CS<F>,
        bytecode: Vec<u64>,
        abi: Vec<u8>,
        compiler: String,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(ARTIFACT_DIGEST_DOMAIN);
        hasher.update(r1cs.digest());
        bincode::serialize_into(&mut hasher, &(&manifest, &bytecode, &abi))
            .expect("hashing cannot fail: the hasher accepts every byte");
        Self {
            compiler,
            manifest,
            r1cs,
            bytecode,
            abi,
            digest: hasher.finalize().into(),
        }
    }

    pub fn field(&self) -> FieldId {
        self.r1cs.field()
    }

    /// SHA-256 over [`ARTIFACT_DIGEST_DOMAIN`], the field-bound [`R1CS::digest`] and the
    /// bincode-encoded manifest, bytecode and ABI. Pin this digest to identify the whole program;
    /// [`SealedR1CS::digest`] identifies only the constraint system.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
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

    pub fn into_parts(self) -> (FieldId, WitnessManifest, SealedR1CS<F>, Vec<u64>, Vec<u8>) {
        (
            self.r1cs.field(),
            self.manifest,
            self.r1cs,
            self.bytecode,
            self.abi,
        )
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, ArtifactError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&ARTIFACT_MAGIC);
        bytes.extend_from_slice(&ARTIFACT_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&field_tag(self.field()).to_le_bytes());
        bytes.extend_from_slice(&self.digest());
        debug_assert_eq!(bytes.len(), HEADER_LEN);
        body_options().serialize_into(
            &mut bytes,
            &BodyRef {
                compiler: &self.compiler,
                manifest: &self.manifest,
                r1cs: self.r1cs.system(),
                bytecode: &self.bytecode,
                abi: &self.abi,
            },
        )?;
        Ok(bytes)
    }

    /// Reads an artifact, verifying its digest and the manifest against the R1CS. Trailing bytes are
    /// an error.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ArtifactError> {
        if bytes.len() < HEADER_LEN || bytes[..ARTIFACT_MAGIC.len()] != ARTIFACT_MAGIC {
            return Err(ArtifactError::MissingHeader);
        }
        let u32_at =
            |start: usize| u32::from_le_bytes(bytes[start..start + 4].try_into().expect("4 bytes"));
        let version = u32_at(VERSION_OFFSET);
        if version != ARTIFACT_FORMAT_VERSION {
            return Err(ArtifactError::UnsupportedVersion(version));
        }
        let field_code = u32_at(FIELD_OFFSET);
        let field = FieldId::from_code(u64::from(field_code))
            .ok_or(ArtifactError::UnknownField(field_code))?;
        if field != F::ID {
            return Err(ArtifactError::WrongField {
                expected: F::ID,
                found: field,
            });
        }
        let digest: [u8; 32] = bytes[DIGEST_OFFSET..HEADER_LEN]
            .try_into()
            .expect("32 bytes");
        let body: Body<F> = body_options().deserialize(&bytes[HEADER_LEN..])?;
        let artifact = Self::new(
            body.manifest,
            SealedR1CS::new(body.r1cs),
            body.bytecode,
            body.abi,
            body.compiler,
        );
        let actual = artifact.digest();
        if actual != digest {
            return Err(ArtifactError::DigestMismatch {
                expected: hex(&digest),
                actual: hex(&actual),
            });
        }
        artifact
            .manifest
            .validate(&artifact.r1cs)
            .map_err(ArtifactError::Invalid)?;
        Ok(artifact)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConstraintsLayout, Field, R1C, WitnessLayout};

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

    fn bytecode() -> Vec<u64> {
        vec![1, 2, 3]
    }

    fn abi() -> Vec<u8> {
        b"{}".to_vec()
    }

    fn build(manifest: WitnessManifest, r1cs: R1CS, bytecode: Vec<u64>, abi: Vec<u8>) -> Artifact {
        Artifact::new(
            manifest,
            SealedR1CS::new(r1cs),
            bytecode,
            abi,
            "test".into(),
        )
    }

    fn artifact(manifest: WitnessManifest, r1cs: R1CS) -> Artifact {
        build(manifest, r1cs, bytecode(), abi())
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
            assert_eq!(&bytes[DIGEST_OFFSET..HEADER_LEN], &decoded.digest());
            assert_ne!(decoded.digest(), decoded.r1cs().digest());
            assert_eq!(decoded.manifest(), &m);
            assert_eq!(decoded.bytecode(), bytecode());
            assert_eq!(decoded.abi(), abi());
            assert_eq!(decoded.compiler(), "test");
            assert_eq!(
                decoded.r1cs().witness_layout.size(),
                r.witness_layout.size()
            );
            assert_eq!(decoded.r1cs().constraints.len(), r.constraints.len());
        }
    }

    #[test]
    fn artifact_rejects_invalid_headers() {
        assert!(matches!(read(&[]), Err(ArtifactError::MissingHeader)));
        let mut bytes = bytes_of(manifest(), r1cs());
        assert!(matches!(
            read(&bytes[..HEADER_LEN - 1]),
            Err(ArtifactError::MissingHeader)
        ));
        for version in [0, ARTIFACT_FORMAT_VERSION - 1, ARTIFACT_FORMAT_VERSION + 1] {
            bytes[VERSION_OFFSET..FIELD_OFFSET].copy_from_slice(&version.to_le_bytes());
            assert!(matches!(
                read(&bytes),
                Err(ArtifactError::UnsupportedVersion(found)) if found == version
            ));
        }
        bytes[VERSION_OFFSET..FIELD_OFFSET].copy_from_slice(&ARTIFACT_FORMAT_VERSION.to_le_bytes());
        bytes[FIELD_OFFSET..DIGEST_OFFSET].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(read(&bytes), Err(ArtifactError::UnknownField(99))));
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
            assert!(
                matches!(round_trip(m, r), Err(ArtifactError::Invalid(_))),
                "overflow case {case}"
            );
        }
    }

    /// The digest is a compatibility promise to consumers that record it: pin its encoding.
    #[test]
    fn digest_encoding_is_pinned() {
        assert_eq!(
            hex(&r1cs().digest()),
            "a1826c2cf819ba7805777209988b33f98071393e14c2671b9ddb3fe9fafe2c52"
        );
        assert_eq!(
            hex(&artifact(manifest(), r1cs()).digest()),
            "ed60cb0a19c5725509c601e780ffb5869f58d6f5d05db0be110f6bf0f4cb4de6"
        );

        // The compiler version stays outside the digest.
        let other = Artifact::new(
            manifest(),
            SealedR1CS::new(r1cs()),
            bytecode(),
            abi(),
            "other".into(),
        );
        assert_eq!(other.digest(), artifact(manifest(), r1cs()).digest());
        assert_eq!(
            read(&other.to_bytes().unwrap()).unwrap().compiler(),
            "other"
        );
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
            forged[DIGEST_OFFSET..HEADER_LEN].copy_from_slice(&honest[DIGEST_OFFSET..HEADER_LEN]);
            forged
        };

        let mut flipped = honest.clone();
        flipped[DIGEST_OFFSET] ^= 1;
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
        rejects(
            &with_honest_digest(with_parts(vec![1, 2, 4], abi())),
            "changed bytecode word",
        );
        rejects(
            &with_honest_digest(with_parts(bytecode(), b"{ }".to_vec())),
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
