use std::fmt::Display;

use ark_ff::{AdditiveGroup, BigInt, PrimeField};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tracing::error;

pub type Field = ark_bn254::Fr;

// ---------------------------------------------------------------------------
// Linear constraint helpers
// ---------------------------------------------------------------------------

pub type LC = Vec<(usize, Field)>;

mod lc_serde {
    use super::*;

    pub fn serialize<S>(lc: &Vec<(usize, ark_bn254::Fr)>, serializer: S) -> Result<S::Ok, S::Error>
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
        return true;
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
        return true;
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

    pub fn need_reference_counting(&self) -> Vec<bool> {
        match self {
            InputValueOrdered::Field(_) => vec![false],
            InputValueOrdered::String(_) => {
                panic!("Strings are not supported in need_reference_counting")
            }
            InputValueOrdered::Vec(_) => vec![true],
            InputValueOrdered::Struct(fields) => {
                let mut reference_counting = vec![];
                for (_field_name, field_value) in fields {
                    reference_counting.extend(field_value.need_reference_counting());
                }
                reference_counting
            }
        }
    }
}
