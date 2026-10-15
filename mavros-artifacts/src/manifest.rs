//! Witness metadata and the versioned `r1cs.bin` format.
//!
//! ```text
//! [ MAVR1CS\0 (8 bytes) ][ format version: u32 LE ][ field: u32 LE ][ artifact digest: 32 bytes ]
//! [ bincode { manifest, r1cs } ]
//! ```
//!
//! The digest binds the field, R1CS and witness manifest; reading verifies it.
//! Bump [`R1CS_ARTIFACT_FORMAT_VERSION`] when changing the header, the serialized manifest, the
//! R1CS or the layouts.
//!
//! The entry blob holds the parameters only, in ABI order. Witness columns follow the wrapper's
//! write order instead: column 0 is the constant one, then the public parameters in ABI order,
//! then the private parameters that survived dead-code elimination in ABI order, and the computed
//! return after `main`'s internal columns.

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::{FieldId, R1CS, SealedR1CS};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ParamVisibility {
    Public,
    Private,
    DataBus,
}

impl ParamVisibility {
    fn is_public(self) -> bool {
        matches!(self, ParamVisibility::Public)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestParam {
    pub name: String,
    pub visibility: ParamVisibility,
    pub blob_offset: usize,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChallengeSlot {
    pub witness_index: usize,
    /// Offset from the start of the post-commitment witness (w2).
    pub w2_offset: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WitnessManifest {
    pub entry_blob_field_count: usize,
    pub parameters: Vec<ManifestParam>,
    pub returns: Option<ReturnSlot>,
    pub challenges: Vec<ChallengeSlot>,
    /// Number of witness values committed before the challenges (the w1/w2 boundary).
    pub pre_commitment_size: usize,
    pub witness_count: usize,
    pub constraint_count: usize,
}

impl WitnessManifest {
    pub fn validate(&self, r1cs: &R1CS) -> Result<(), String> {
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
        let pre_commitment_size = add(
            layout.algebraic_size,
            layout.multiplicities_size,
            "pre-commitment size",
        )?;
        let witness_count = [
            layout.challenges_size,
            layout.tables_data_size,
            layout.lookups_data_size,
        ]
        .into_iter()
        .try_fold(pre_commitment_size, |size, len| {
            add(size, len, "witness count")
        })?;
        check(self.witness_count == witness_count, "witness count")?;
        check(
            self.constraint_count == r1cs.constraints.len(),
            "constraint count",
        )?;
        check(
            self.pre_commitment_size == pre_commitment_size,
            "pre-commitment size",
        )?;
        check(
            self.challenges.len() == layout.challenges_size,
            "challenge count",
        )?;
        for (i, challenge) in self.challenges.iter().enumerate() {
            check(
                challenge.w2_offset == i
                    && challenge.witness_index
                        == add(pre_commitment_size, i, "challenge position")?,
                "challenge positions",
            )?;
        }

        let mut blob_offset = 0;
        for param in &self.parameters {
            check(param.blob_offset == blob_offset, "parameter blob offset")?;
            blob_offset = add(blob_offset, param.width, "entry blob")?;
        }
        check(
            self.entry_blob_field_count == blob_offset,
            "entry blob field count",
        )?;

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

pub const R1CS_ARTIFACT_MAGIC: [u8; 8] = *b"MAVR1CS\0";
/// Format version this crate writes and the only one it reads.
pub const R1CS_ARTIFACT_FORMAT_VERSION: u32 = 1;

/// Domain separator of [`R1CSArtifact::digest`]. Bump when its preimage encoding changes.
pub const R1CS_ARTIFACT_DIGEST_DOMAIN: &[u8] = b"mavros-r1cs-artifact-v1\0";

const VERSION_OFFSET: usize = R1CS_ARTIFACT_MAGIC.len();
const FIELD_OFFSET: usize = VERSION_OFFSET + 4;
const DIGEST_OFFSET: usize = FIELD_OFFSET + 4;
const HEADER_LEN: usize = DIGEST_OFFSET + 32;

#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error("r1cs.bin does not start with the MAVR1CS header, so it is not a Mavros R1CS artifact")]
    MissingHeader,
    #[error("unsupported r1cs.bin format version {0}; expected {R1CS_ARTIFACT_FORMAT_VERSION}")]
    UnsupportedVersion(u32),
    #[error("r1cs.bin names unknown field {0}")]
    UnknownField(u32),
    #[error("r1cs.bin body: {0}")]
    Body(#[from] bincode::Error),
    #[error(
        "r1cs.bin digest mismatch: the header says {expected} but the artifact hashes to {actual}"
    )]
    DigestMismatch { expected: String, actual: String },
    #[error("{0}")]
    Invalid(String),
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Debug, Deserialize)]
struct Body {
    manifest: WitnessManifest,
    r1cs: R1CS,
}

#[derive(Serialize)]
struct BodyRef<'a> {
    manifest: &'a WitnessManifest,
    r1cs: &'a R1CS,
}

#[derive(Debug)]
pub struct R1CSArtifact {
    manifest: WitnessManifest,
    r1cs: SealedR1CS,
    digest: [u8; 32],
}

impl R1CSArtifact {
    pub fn new(manifest: WitnessManifest, r1cs: SealedR1CS) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(R1CS_ARTIFACT_DIGEST_DOMAIN);
        hasher.update(r1cs.digest());
        bincode::serialize_into(&mut hasher, &manifest)
            .expect("hashing a witness manifest cannot fail: the hasher accepts every byte");
        Self {
            manifest,
            r1cs,
            digest: hasher.finalize().into(),
        }
    }

    pub fn field(&self) -> FieldId {
        self.r1cs.field()
    }

    /// SHA-256 over [`R1CS_ARTIFACT_DIGEST_DOMAIN`], the field-bound [`R1CS::digest`] and the
    /// bincode-encoded manifest. Pin this digest to identify both the system and its public
    /// input/output mapping; [`SealedR1CS::digest`] identifies only the constraint system.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn manifest(&self) -> &WitnessManifest {
        &self.manifest
    }

    pub fn r1cs(&self) -> &SealedR1CS {
        &self.r1cs
    }

    pub fn into_parts(self) -> (FieldId, WitnessManifest, SealedR1CS) {
        (self.r1cs.field(), self.manifest, self.r1cs)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, ArtifactError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&R1CS_ARTIFACT_MAGIC);
        bytes.extend_from_slice(&R1CS_ARTIFACT_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.field().code().to_le_bytes());
        bytes.extend_from_slice(&self.digest());
        debug_assert_eq!(bytes.len(), HEADER_LEN);
        bincode::serialize_into(
            &mut bytes,
            &BodyRef {
                manifest: &self.manifest,
                r1cs: self.r1cs.system(),
            },
        )?;
        Ok(bytes)
    }

    /// Reads an artifact, verifying its digest and the manifest against the R1CS.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ArtifactError> {
        if bytes.len() < HEADER_LEN || bytes[..R1CS_ARTIFACT_MAGIC.len()] != R1CS_ARTIFACT_MAGIC {
            return Err(ArtifactError::MissingHeader);
        }
        let u32_at =
            |start: usize| u32::from_le_bytes(bytes[start..start + 4].try_into().expect("4 bytes"));
        let version = u32_at(VERSION_OFFSET);
        if version != R1CS_ARTIFACT_FORMAT_VERSION {
            return Err(ArtifactError::UnsupportedVersion(version));
        }
        let field_code = u32_at(FIELD_OFFSET);
        let field =
            FieldId::from_code(field_code).ok_or(ArtifactError::UnknownField(field_code))?;
        let digest: [u8; 32] = bytes[DIGEST_OFFSET..HEADER_LEN]
            .try_into()
            .expect("32 bytes");
        let body: Body = bincode::deserialize(&bytes[HEADER_LEN..])?;
        let artifact = Self::new(body.manifest, SealedR1CS::new(body.r1cs, field));
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
            entry_blob_field_count: 2,
            parameters: vec![
                ManifestParam {
                    name: "x".into(),
                    visibility: ParamVisibility::Private,
                    blob_offset: 0,
                    width: 1,
                    column_start: Some(2),
                },
                ManifestParam {
                    name: "y".into(),
                    visibility: ParamVisibility::Public,
                    blob_offset: 1,
                    width: 1,
                    column_start: Some(1),
                },
            ],
            returns: Some(ReturnSlot {
                column_start: Some(5),
                len: 1,
                visibility: ParamVisibility::Public,
            }),
            challenges: vec![],
            pre_commitment_size: 6,
            witness_count: 6,
            constraint_count: 1,
        }
    }

    fn bytes_of(manifest: WitnessManifest, r1cs: R1CS) -> Vec<u8> {
        R1CSArtifact::new(manifest, SealedR1CS::new(r1cs, FieldId::Bn254))
            .to_bytes()
            .unwrap()
    }

    fn round_trip(manifest: WitnessManifest, r1cs: R1CS) -> Result<R1CSArtifact, ArtifactError> {
        R1CSArtifact::from_bytes(&bytes_of(manifest, r1cs))
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
                m.parameters[1].blob_offset = 0;
                m.entry_blob_field_count = 1;
            },
        ];
        for mutate in cases {
            let (mut m, mut r) = (manifest(), r1cs());
            mutate(&mut m, &mut r);
            let bytes = bytes_of(m.clone(), r.clone());
            assert!(bytes.starts_with(&R1CS_ARTIFACT_MAGIC));
            assert_eq!(
                bytes,
                bytes_of(m.clone(), r.clone()),
                "serialization is deterministic"
            );
            let decoded = R1CSArtifact::from_bytes(&bytes).unwrap();
            assert_eq!(decoded.field(), FieldId::Bn254);
            assert_eq!(decoded.r1cs().digest(), r.digest(FieldId::Bn254));
            assert_eq!(&bytes[DIGEST_OFFSET..HEADER_LEN], &decoded.digest());
            assert_ne!(decoded.digest(), decoded.r1cs().digest());
            assert_eq!(decoded.manifest(), &m);
            assert_eq!(
                decoded.r1cs().witness_layout.size(),
                r.witness_layout.size()
            );
            assert_eq!(decoded.r1cs().constraints.len(), r.constraints.len());
        }
    }

    #[test]
    fn artifact_rejects_invalid_headers() {
        assert!(matches!(
            R1CSArtifact::from_bytes(&[]),
            Err(ArtifactError::MissingHeader)
        ));
        let mut bytes = bytes_of(manifest(), r1cs());
        assert!(matches!(
            R1CSArtifact::from_bytes(&bytes[..HEADER_LEN - 1]),
            Err(ArtifactError::MissingHeader)
        ));
        for version in [0, R1CS_ARTIFACT_FORMAT_VERSION + 1] {
            bytes[VERSION_OFFSET..FIELD_OFFSET].copy_from_slice(&version.to_le_bytes());
            assert!(matches!(
                R1CSArtifact::from_bytes(&bytes),
                Err(ArtifactError::UnsupportedVersion(found)) if found == version
            ));
        }
        bytes[VERSION_OFFSET..FIELD_OFFSET]
            .copy_from_slice(&R1CS_ARTIFACT_FORMAT_VERSION.to_le_bytes());
        bytes[FIELD_OFFSET..DIGEST_OFFSET].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(
            R1CSArtifact::from_bytes(&bytes),
            Err(ArtifactError::UnknownField(99))
        ));
    }

    #[test]
    fn artifact_rejects_inconsistent_entry_slots() {
        let cases: [fn(&mut WitnessManifest, &mut R1CS); 10] = [
            |m, _| m.parameters[1].blob_offset = 900,
            |m, _| m.entry_blob_field_count = 3,
            // A public parameter always has columns, and they come first.
            |m, _| m.parameters[1].column_start = None,
            |m, _| m.parameters[1].column_start = Some(2),
            |m, _| m.parameters[0].column_start = Some(1),
            |m, _| m.parameters[0].column_start = Some(3),
            // The return follows the parameters and stays inside the algebraic section.
            |m, _| m.returns.as_mut().unwrap().column_start = None,
            |m, _| m.returns.as_mut().unwrap().column_start = Some(2),
            |m, _| m.returns.as_mut().unwrap().len = 2,
            |m, r| {
                m.pre_commitment_size = 2;
                m.witness_count = 2;
                r.witness_layout.algebraic_size = 2;
            },
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
        let cases: [fn(&mut WitnessManifest, &mut R1CS); 4] = [
            |m, _| m.parameters[0].width = usize::MAX,
            |_, r| r.witness_layout.multiplicities_size = usize::MAX,
            |_, r| r.witness_layout.lookups_data_size = usize::MAX,
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

    #[test]
    fn challenges_must_sit_at_the_layout_positions() {
        let (mut m, mut r) = (manifest(), r1cs());
        r.witness_layout.multiplicities_size = 2;
        r.witness_layout.challenges_size = 2;
        r.witness_layout.tables_data_size = 3;
        r.witness_layout.lookups_data_size = 4;
        m.pre_commitment_size = 8;
        m.witness_count = 17;
        m.challenges = vec![
            ChallengeSlot {
                witness_index: 8,
                w2_offset: 0,
            },
            ChallengeSlot {
                witness_index: 9,
                w2_offset: 1,
            },
        ];
        assert!(round_trip(m.clone(), r.clone()).is_ok());
        m.challenges[1].w2_offset = 0;
        assert!(matches!(round_trip(m, r), Err(ArtifactError::Invalid(_))));
    }

    /// The digest is a compatibility promise to consumers that record it: pin its encoding.
    #[test]
    fn digest_encoding_is_pinned() {
        assert_eq!(
            hex(&r1cs().digest(FieldId::Bn254)),
            "a00d2ffcd12de9ef9b7642233971e38e1bb2edd7771436a9e85412471d1bb795"
        );
        let artifact = R1CSArtifact::new(manifest(), SealedR1CS::new(r1cs(), FieldId::Bn254));
        assert_eq!(
            hex(&artifact.digest()),
            "dc87920b5ff99db926f5baf53d52421e03c5870fecb94fb72a3a36672aaef7c8"
        );
    }

    /// A flipped header byte, another R1CS under the honest digest, or a manifest edit that
    /// still validates: each is a digest mismatch.
    #[test]
    fn tampering_is_rejected() {
        let honest = bytes_of(manifest(), r1cs());
        let rejects = |bytes: &[u8], case: &str| {
            assert!(
                matches!(
                    R1CSArtifact::from_bytes(bytes),
                    Err(ArtifactError::DigestMismatch { .. })
                ),
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
        assert!(matches!(
            R1CSArtifact::from_bytes(&bytes),
            Err(ArtifactError::Body(_))
        ));
    }
}
