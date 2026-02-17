pub mod array;
pub mod bytecode;
pub mod interpreter;

pub type Field = ark_bn254::Fr;

use serde::{Deserialize, Serialize};

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
