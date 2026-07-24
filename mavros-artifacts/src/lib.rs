use std::{collections::BTreeMap, fmt::Display};

use ark_ff::{AdditiveGroup, BigInt, PrimeField};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tracing::error;

// FIELD-ASSUMPTION: L1-alias
pub type Field = ark_bn254::Fr;

const MAX_TIMELINE_SAMPLES: usize = 100_000;

/// Aggregated stacks in the folded format consumed by Brendan Gregg's
/// `flamegraph.pl`, plus a bounded chronological sample stream.
///
/// Frames are stored root-first and weights are deterministic integer units
/// (for example constraints, witnesses, or simulated VM instructions).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlamegraphProfile {
    stacks: BTreeMap<Vec<String>, StackProfile>,
    sample_stacks: Vec<Vec<String>>,
    timeline_samples: Vec<TimelineSample>,
    sample_interval: u64,
    next_block_start: u64,
    next_sample_at: u64,
    sampler_state: u64,
    total_weight: u64,
}

/// An interned stack in a [`FlamegraphProfile`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlamegraphStackId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StackProfile {
    weight: u64,
    sample_id: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TimelineSample {
    position: u64,
    stack_id: usize,
}

impl Default for FlamegraphProfile {
    fn default() -> Self {
        Self {
            stacks: BTreeMap::new(),
            sample_stacks: Vec::new(),
            timeline_samples: Vec::new(),
            sample_interval: 1,
            next_block_start: 0,
            next_sample_at: 0,
            sampler_state: 0x9e37_79b9_7f4a_7c15,
            total_weight: 0,
        }
    }
}

impl FlamegraphProfile {
    pub fn record<I, S>(&mut self, stack: I, weight: u64)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        if weight == 0 {
            return;
        }
        let Some(stack_id) = self.intern_stack(stack) else {
            return;
        };
        self.record_interned(stack_id, weight);
    }

    /// Intern a root-first call stack so repeated samples can record it without
    /// cloning its frame names.
    pub fn intern_stack<I, S>(&mut self, stack: I) -> Option<FlamegraphStackId>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let stack = stack.into_iter().map(Into::into).collect::<Vec<_>>();
        if stack.is_empty() {
            return None;
        }
        let next_sample_id = self.sample_stacks.len();
        let sample_id = match self.stacks.entry(stack) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.get().sample_id,
            std::collections::btree_map::Entry::Vacant(entry) => {
                self.sample_stacks.push(entry.key().clone());
                entry.insert(StackProfile {
                    weight: 0,
                    sample_id: next_sample_id,
                });
                next_sample_id
            }
        };
        Some(FlamegraphStackId(sample_id))
    }

    /// Record weight against an already-interned stack.
    pub fn record_interned(&mut self, stack_id: FlamegraphStackId, weight: u64) {
        if weight == 0 {
            return;
        }
        let stack = self
            .sample_stacks
            .get(stack_id.0)
            .expect("FlameGraph stack id belongs to this profile");
        self.stacks
            .get_mut(stack.as_slice())
            .expect("interned FlameGraph stack is indexed")
            .weight += weight;

        let segment_end = self.total_weight + weight;
        while self.next_sample_at < segment_end {
            if self.timeline_samples.len() == MAX_TIMELINE_SAMPLES {
                self.compact_timeline(self.next_sample_at);
                continue;
            }
            let sample_position = self.next_sample_at;
            self.timeline_samples.push(TimelineSample {
                position: sample_position,
                stack_id: stack_id.0,
            });
            self.schedule_next_sample();
        }
        self.total_weight = segment_end;
    }

    pub fn total_weight(&self) -> u64 {
        self.total_weight
    }

    pub fn is_empty(&self) -> bool {
        self.total_weight == 0
    }

    pub fn stacks(&self) -> impl Iterator<Item = (&[String], u64)> {
        self.stacks
            .iter()
            .filter(|(_, profile)| profile.weight > 0)
            .map(|(stack, profile)| (stack.as_slice(), profile.weight))
    }

    /// Deterministic root-first call-stack samples in execution order.
    pub fn timeline_samples(&self) -> impl Iterator<Item = (u64, &[String])> {
        self.timeline_samples.iter().map(|sample| {
            (
                sample.position,
                self.sample_stacks[sample.stack_id].as_slice(),
            )
        })
    }

    fn compact_timeline(&mut self, cursor: u64) {
        let previous_samples = std::mem::take(&mut self.timeline_samples);
        self.timeline_samples.reserve(previous_samples.len() / 2);
        for pair in previous_samples.chunks_exact(2) {
            let selected = if self.next_random() & 1 == 0 {
                pair[0]
            } else {
                pair[1]
            };
            self.timeline_samples.push(selected);
        }
        self.sample_interval *= 2;
        self.next_block_start = cursor.div_ceil(self.sample_interval) * self.sample_interval;
        self.next_sample_at = self.next_block_start + self.random_sample_offset();
    }

    fn schedule_next_sample(&mut self) {
        self.next_block_start += self.sample_interval;
        self.next_sample_at = self.next_block_start + self.random_sample_offset();
    }

    fn random_sample_offset(&mut self) -> u64 {
        if self.sample_interval == 1 {
            0
        } else {
            self.next_random() % self.sample_interval
        }
    }

    fn next_random(&mut self) -> u64 {
        self.sampler_state ^= self.sampler_state >> 12;
        self.sampler_state ^= self.sampler_state << 25;
        self.sampler_state ^= self.sampler_state >> 27;
        self.sampler_state = self.sampler_state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        self.sampler_state
    }

    /// Render the already-collapsed input expected by `flamegraph.pl`.
    pub fn to_folded(&self) -> String {
        let mut folded = String::new();
        for (stack, profile) in &self.stacks {
            if profile.weight == 0 {
                continue;
            }
            let frames = stack
                .iter()
                .map(|frame| sanitize_flamegraph_frame(frame))
                .collect::<Vec<_>>()
                .join(";");
            folded.push_str(&frames);
            folded.push(' ');
            folded.push_str(&profile.weight.to_string());
            folded.push('\n');
        }
        folded
    }
}

fn sanitize_flamegraph_frame(frame: &str) -> String {
    frame
        .chars()
        .map(|character| match character {
            ';' => ':',
            '\n' | '\r' => ' ',
            character => character,
        })
        .collect()
}

#[cfg(test)]
mod flamegraph_profile_tests {
    use super::FlamegraphProfile;

    #[test]
    fn folded_profiles_are_aggregated_sorted_and_sanitized() {
        let mut profile = FlamegraphProfile::default();
        profile.record(["main", "z;helper"], 2);
        profile.record(["main", "z;helper"], 3);
        profile.record(["main", "a\nhelper"], 1);
        profile.record(["main", "z;helper"], 4);

        assert_eq!(profile.total_weight(), 10);
        assert_eq!(profile.to_folded(), "main;a helper 1\nmain;z:helper 9\n");
        assert_eq!(profile.sample_interval, 1);
        assert_eq!(
            profile
                .timeline_samples()
                .map(|(_, stack)| stack.last().unwrap().as_str())
                .collect::<Vec<_>>(),
            vec![
                "z;helper",
                "z;helper",
                "z;helper",
                "z;helper",
                "z;helper",
                "a\nhelper",
                "z;helper",
                "z;helper",
                "z;helper",
                "z;helper",
            ]
        );
    }

    #[test]
    fn timeline_sampling_is_bounded_and_deterministic() {
        let mut profile = FlamegraphProfile::default();
        profile.record(["main"], 250_001);
        let mut repeated_profile = FlamegraphProfile::default();
        repeated_profile.record(["main"], 250_001);

        assert_eq!(profile.total_weight(), 250_001);
        assert_eq!(profile, repeated_profile);
        assert_eq!(profile.sample_interval, 4);
        assert!((62_499..=62_501).contains(&profile.timeline_samples().count()));
        assert!(
            profile
                .timeline_samples()
                .all(|(_, stack)| stack == ["main".to_string()])
        );
        assert!(
            profile
                .timeline_samples()
                .map(|(position, _)| position)
                .is_sorted()
        );
    }

    #[test]
    fn exactly_the_sample_limit_keeps_every_instruction() {
        let mut profile = FlamegraphProfile::default();
        profile.record(["main"], super::MAX_TIMELINE_SAMPLES as u64);

        assert_eq!(
            profile.timeline_samples().count(),
            super::MAX_TIMELINE_SAMPLES
        );
        assert_eq!(profile.sample_interval, 1);

        profile.record(["main"], 1);
        assert!(profile.timeline_samples().count() < super::MAX_TIMELINE_SAMPLES);
        assert_eq!(profile.sample_interval, 2);
    }

    #[test]
    fn interned_stacks_record_without_reinterning_names() {
        let mut profile = FlamegraphProfile::default();
        let stack = profile.intern_stack(["main", "helper"]).unwrap();
        profile.record_interned(stack, 2);
        profile.record_interned(stack, 3);

        assert_eq!(profile.to_folded(), "main;helper 5\n");
    }
}

// ---------------------------------------------------------------------------
// Linear constraint helpers
// ---------------------------------------------------------------------------

pub type LC = Vec<(usize, Field)>;

mod lc_serde {
    use super::*;

    // FIELD-ASSUMPTION: L1-serde
    pub fn serialize<S>(lc: &[(usize, ark_bn254::Fr)], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let converted: Vec<(usize, [u64; 4])> = lc
            .iter()
            .map(|(idx, coeff)| (*idx, coeff.into_bigint().0))
            .collect();
        converted.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<(usize, ark_bn254::Fr)>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let converted: Vec<(usize, [u64; 4])> = Deserialize::deserialize(deserializer)?;
        Ok(converted
            .into_iter()
            .map(|(idx, limbs)| {
                (
                    idx,
                    ark_bn254::Fr::from_bigint(BigInt(limbs)).expect("Invalid field element"),
                )
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// R1C – a single Rank-1 Constraint
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R1C {
    #[serde(with = "lc_serde")]
    pub a: LC,
    #[serde(with = "lc_serde")]
    pub b: LC,
    #[serde(with = "lc_serde")]
    pub c: LC,
}

// FIELD-ASSUMPTION: L4-sign
fn field_to_string(c: ark_bn254::Fr) -> String {
    if c.into_bigint() > Field::MODULUS_MINUS_ONE_DIV_TWO {
        format!("-{}", -c)
    } else {
        c.to_string()
    }
}

impl Display for R1C {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let a_str = self
            .a
            .iter()
            .map(|(i, c)| format!("{} * v{}", field_to_string(*c), i))
            .collect::<Vec<_>>()
            .join(" + ");
        let b_str = self
            .b
            .iter()
            .map(|(i, c)| format!("{} * v{}", field_to_string(*c), i))
            .collect::<Vec<_>>()
            .join(" + ");
        let c_str = self
            .c
            .iter()
            .map(|(i, c)| format!("{} * v{}", field_to_string(*c), i))
            .collect::<Vec<_>>()
            .join(" + ");

        write!(f, "({}) * ({}) - ({}) = 0", a_str, b_str, c_str)
    }
}

// ---------------------------------------------------------------------------
// Lookup table layout
// ---------------------------------------------------------------------------

/// How a lookup table's allocation entries are laid out in the constraint
/// system. This is the discriminant Phase 2 of witness generation branches on
/// to fill each entry's constraint(s); it is carried through the witgen VM's
/// table-info slot as the integer [`TableKind::code`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableKind {
    /// Key-only (rangecheck): one constraint per entry, `denom = α - i`.
    RangeCheck,
    /// Key-value with witness values: two constraints per entry
    /// (`β·v = -x`; `y·(α - i - x) = m`).
    Array,
    /// Key-value whose values are the compile-time constants `spread(i)`. Both
    /// operands of every entry are constant, so `β·spread(i)` folds into the
    /// denominator and each entry needs only one constraint
    /// (`y·(α - i + β·spread(i)) = m`).
    Spread,
}

impl TableKind {
    /// Wire encoding stored in the table-info slot.
    pub fn code(self) -> u32 {
        match self {
            TableKind::RangeCheck => 0,
            TableKind::Array => 1,
            TableKind::Spread => 2,
        }
    }

    pub fn from_code(code: u32) -> Self {
        match code {
            0 => TableKind::RangeCheck,
            1 => TableKind::Array,
            2 => TableKind::Spread,
            other => panic!("invalid TableKind code: {other}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Witness & constraints layout
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub struct WitnessLayout {
    pub algebraic_size: usize,
    pub multiplicities_size: usize,

    pub challenges_size: usize,

    pub tables_data_size: usize,
    pub lookups_data_size: usize,
}

impl WitnessLayout {
    pub fn algebraic_start(&self) -> usize {
        0
    }

    pub fn algebraic_end(&self) -> usize {
        self.algebraic_size
    }

    pub fn multiplicities_start(&self) -> usize {
        self.algebraic_end()
    }

    pub fn multiplicities_end(&self) -> usize {
        self.multiplicities_start() + self.multiplicities_size
    }

    pub fn challenges_start(&self) -> usize {
        self.multiplicities_end()
    }

    pub fn challenges_end(&self) -> usize {
        self.challenges_start() + self.challenges_size
    }

    pub fn next_challenge(&mut self) -> usize {
        let challenge_id = self.challenges_end();
        self.challenges_size += 1;
        challenge_id
    }

    pub fn tables_data_start(&self) -> usize {
        self.challenges_end()
    }

    pub fn tables_data_end(&self) -> usize {
        self.tables_data_size + self.tables_data_start()
    }

    pub fn next_table_data(&mut self) -> usize {
        let table_data_id = self.tables_data_end();
        self.tables_data_size += 1;
        table_data_id
    }

    pub fn lookups_data_start(&self) -> usize {
        self.tables_data_end()
    }

    pub fn lookups_data_end(&self) -> usize {
        self.lookups_data_size + self.lookups_data_start()
    }

    pub fn next_lookups_data(&mut self) -> usize {
        let lookups_data_id = self.lookups_data_end();
        self.lookups_data_size += 1;
        lookups_data_id
    }

    pub fn size(&self) -> usize {
        self.algebraic_size
            + self.multiplicities_size
            + self.challenges_size
            + self.tables_data_size
            + self.lookups_data_size
    }

    pub fn pre_commitment_size(&self) -> usize {
        self.algebraic_size + self.multiplicities_size
    }

    pub fn post_commitment_size(&self) -> usize {
        self.challenges_size + self.tables_data_size + self.lookups_data_size
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub struct ConstraintsLayout {
    pub algebraic_size: usize,
    pub tables_data_size: usize,
    pub lookups_data_size: usize,
}

impl ConstraintsLayout {
    pub fn size(&self) -> usize {
        self.algebraic_size + self.tables_data_size + self.lookups_data_size
    }

    pub fn tables_data_start(&self) -> usize {
        self.algebraic_size
    }

    pub fn lookups_data_start(&self) -> usize {
        self.algebraic_size + self.tables_data_size
    }
}

// ---------------------------------------------------------------------------
// R1CS – the full constraint system
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R1CS {
    pub witness_layout: WitnessLayout,
    pub constraints_layout: ConstraintsLayout,
    pub constraints: Vec<R1C>,
}

impl R1CS {
    pub fn compute_derivatives(
        &self,
        coeffs: &[Field],
        res_a: &mut [Field],
        res_b: &mut [Field],
        res_c: &mut [Field],
    ) {
        for (r1c, coeff) in self.constraints.iter().zip(coeffs.iter()) {
            for (a_ix, a_coeff) in r1c.a.iter() {
                res_a[*a_ix] += *a_coeff * *coeff;
            }
            for (b_ix, b_coeff) in r1c.b.iter() {
                res_b[*b_ix] += *b_coeff * *coeff;
            }
            for (c_ix, c_coeff) in r1c.c.iter() {
                res_c[*c_ix] += *c_coeff * *coeff;
            }
        }
    }

    pub fn check_witgen_output(
        &self,
        pre_comm_witness: &[Field],
        post_comm_witness: &[Field],
        a: &[Field],
        b: &[Field],
        c: &[Field],
    ) -> bool {
        let witness = [pre_comm_witness, post_comm_witness].concat();
        if a.len() != self.constraints_layout.size() {
            error!(message = %"The a vector has the wrong length", expected = self.constraints_layout.size(), actual = a.len());
            return false;
        }
        if b.len() != self.constraints_layout.size() {
            error!(message = %"The b vector has the wrong length", expected = self.constraints_layout.size(), actual = b.len());
            return false;
        }
        if c.len() != self.constraints_layout.size() {
            error!(message = %"The c vector has the wrong length", expected = self.constraints_layout.size(), actual = c.len());
            return false;
        }
        for (i, r1c) in self.constraints.iter().enumerate() {
            let av = r1c
                .a
                .iter()
                .map(|(i, c)| c * &witness[*i])
                .sum::<ark_bn254::Fr>();

            let bv = r1c
                .b
                .iter()
                .map(|(i, c)| c * &witness[*i])
                .sum::<ark_bn254::Fr>();

            let cv = r1c
                .c
                .iter()
                .map(|(i, c)| c * &witness[*i])
                .sum::<ark_bn254::Fr>();
            let mut fail = false;
            if av * bv != cv {
                error!(message = %"R1CS constraint failed to verify", index = i);
                fail = true;
            }
            if av != a[i] {
                error!(message = %"Wrong A value for constraint", index = i, actual = a[i].to_string(), expected = av.to_string());
                fail = true;
            }
            if bv != b[i] {
                error!(message = %"Wrong B value for constraint", index = i, actual = b[i].to_string(), expected = bv.to_string());
                fail = true;
            }
            if cv != c[i] {
                error!(message = %"Wrong C value for constraint", index = i, actual = c[i].to_string(), expected = cv.to_string());
                fail = true;
            }
            if fail {
                return false;
            }
        }
        true
    }

    pub fn check_ad_output(&self, coeffs: &[Field], a: &[Field], b: &[Field], c: &[Field]) -> bool {
        let mut a = a.to_vec();
        let mut b = b.to_vec();
        let mut c = c.to_vec();
        for (r1c, coeff) in self.constraints.iter().zip(coeffs.iter()) {
            for (a_ix, a_coeff) in r1c.a.iter() {
                a[*a_ix] -= *a_coeff * *coeff;
            }
            for (b_ix, b_coeff) in r1c.b.iter() {
                b[*b_ix] -= *b_coeff * *coeff;
            }
            for (c_ix, c_coeff) in r1c.c.iter() {
                c[*c_ix] -= *c_coeff * *coeff;
            }
        }
        let mut wrongs = 0;
        for i in 0..a.len() {
            if a[i] != Field::ZERO {
                if wrongs == 0 {
                    error!(message = %"Wrong A deriv for witness", index = i);
                }
                wrongs += 1;
            }
            if b[i] != Field::ZERO {
                if wrongs == 0 {
                    error!(message = %"Wrong B deriv for witness", index = i);
                }
                wrongs += 1;
            }
            if c[i] != Field::ZERO {
                if wrongs == 0 {
                    error!(message = %"Wrong C deriv for witness", index = i);
                }
                wrongs += 1;
            }
        }
        if wrongs > 0 {
            error!("{} out of {} wrong derivatives", wrongs, 3 * a.len());
            return false;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// InputValueOrdered – ABI-ordered input representation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum InputValueOrdered {
    Field(Field),
    String(String),
    Vec(Vec<InputValueOrdered>),
    Struct(Vec<(String, InputValueOrdered)>),
}

impl InputValueOrdered {
    pub fn field_sizes(&self) -> Vec<usize> {
        match self {
            // FIELD-ASSUMPTION: L1-serde
            InputValueOrdered::Field(_) => vec![4],
            InputValueOrdered::String(_) => panic!("Strings are not supported in element_size"),
            InputValueOrdered::Vec(_) => vec![1],
            InputValueOrdered::Struct(fields) => {
                let mut total_size = vec![];
                for (_field_name, field_value) in fields {
                    total_size.extend(field_value.field_sizes());
                }
                total_size
            }
        }
    }
}
