//! Forward evaluation of the constraint program at an arbitrary witness vector.
//!
//! Witness references are field values in this interpretation. No AD nodes or
//! matrix rows are constructed. Lookup rows use the same layout as the R1CS.

use mavros_artifacts::MavrosField;

use crate::{
    ConstraintsLayout, LogupStrides, TableKind, WitnessLayout,
    bytecode::{TableInfo, Tape},
};

/// Borrowed execution buffers, installed only after the runner checks their lengths.
/// All pointers remain valid until dispatch finishes. Only verifier opcodes may
/// access this member of the VM's execution-data union.
#[derive(Clone, Copy)]
pub(crate) struct VerifierArrays {
    pub outputs: [Tape; 3],
    pub witness: Tape,
    pub coeffs: Tape,
    pub witness_cursor: usize,
    pub constraint_cursor: usize,
    /// The first repetition's `alpha`.
    challenge: usize,
    /// The table and lookup cursors below run over repetition 0; repetition `k` of a row sits `k`
    /// strides past it.
    logup: LogupStrides,
    multiplicities: usize,
    table_witness: usize,
    table_constraint: usize,
    lookup_witness: usize,
    lookup_constraint: usize,
}

impl VerifierArrays {
    pub fn new<F: MavrosField>(
        outputs: [*mut F; 3],
        witness: *const F,
        coeffs: *const F,
        wl: WitnessLayout,
        cl: ConstraintsLayout,
    ) -> Self {
        Self {
            outputs: outputs.map(Tape::of),
            witness: Tape::of(witness as *mut F),
            coeffs: Tape::of(coeffs as *mut F),
            witness_cursor: 0,
            constraint_cursor: 0,
            challenge: wl.challenges_start(),
            logup: LogupStrides::new(&wl, &cl),
            multiplicities: wl.multiplicities_start(),
            table_witness: wl.tables_data_start(),
            table_constraint: cl.tables_data_start(),
            lookup_witness: wl.lookups_data_start(),
            lookup_constraint: cl.lookups_data_start(),
        }
    }

    pub unsafe fn w<F: MavrosField>(&self, index: usize) -> F {
        unsafe { *self.witness.at::<F>(index) }
    }
    unsafe fn r<F: MavrosField>(&self, index: usize) -> F {
        unsafe { *self.coeffs.at::<F>(index) }
    }

    /// Accumulate one row's three forward-evaluated linear combinations.
    unsafe fn row<F: MavrosField>(&mut self, index: usize, a: F, b: F, c: F) {
        unsafe {
            let r = self.r::<F>(index);
            for (output, value) in self.outputs.iter().zip([a, b, c]) {
                *output.slot::<F>() += r * value;
            }
        }
    }

    /// The value of repetition `repetition`'s `alpha`.
    unsafe fn alpha<F: MavrosField>(&self, repetition: usize) -> F {
        unsafe { self.w::<F>(self.challenge + self.logup.alpha(repetition)) }
    }

    /// The value of repetition `repetition`'s `beta`, which only a layout with a value-column table
    /// has.
    unsafe fn beta<F: MavrosField>(&self, repetition: usize) -> F {
        unsafe { self.w::<F>(self.challenge + self.logup.beta(repetition)) }
    }

    /// The witness and constraint offsets of `table`'s repetition `repetition`.
    fn table_offsets(&self, table: &TableInfo, repetition: usize) -> (usize, usize) {
        (
            table.elem_inverses_witness_section_offset + repetition * self.logup.witness_tables,
            table.elem_inverses_constraint_section_offset
                + repetition * self.logup.constraint_tables,
        )
    }

    /// The next table's row positions, before its length is known.
    pub fn begin_table(&self, kind: TableKind) -> (TableInfo, usize) {
        (
            TableInfo {
                multiplicities_wit: Tape::NONE,
                kind,
                length: 0,
                elem_inverses_witness_section_offset: self.table_witness,
                elem_inverses_constraint_section_offset: self.table_constraint,
            },
            self.multiplicities,
        )
    }

    pub fn reserve_table(&mut self, table: &TableInfo) {
        let width = if table.kind == TableKind::Array { 2 } else { 1 };
        self.multiplicities += table.length;
        self.table_witness += width * table.length;
        self.table_constraint += width * table.length + 1;
    }

    /// Reserve a table in the deterministic order used by matrix generation.
    pub fn table(&mut self, kind: TableKind, length: usize) -> (TableInfo, usize) {
        let (mut table, multiplicities) = self.begin_table(kind);
        table.length = length;
        self.reserve_table(&table);
        (table, multiplicities)
    }

    /// Range, spread and pow2 tables have constant entries and one folded row per entry.
    pub unsafe fn constant_table<F: MavrosField>(
        &mut self,
        table: &TableInfo,
        multiplicities: usize,
        values: impl Iterator<Item = F> + Clone,
    ) {
        unsafe {
            let w0 = self.w::<F>(0);
            for repetition in 0..self.logup.repetitions {
                let (witnesses, constraints) = self.table_offsets(table, repetition);
                let mut sum = F::ZERO;
                let alpha = self.alpha::<F>(repetition);
                // Range tables need no beta column.
                let beta = if table.kind == TableKind::RangeCheck {
                    F::ZERO
                } else {
                    self.beta::<F>(repetition)
                };
                for (i, value) in values.clone().enumerate() {
                    let y = self.w::<F>(witnesses + i);
                    self.row(
                        constraints + i,
                        y,
                        alpha - F::from(i as u64) * w0 + beta * value,
                        self.w::<F>(multiplicities + i),
                    );
                    sum += y;
                }
                self.row(constraints + table.length, sum, w0, F::ZERO);
            }
        }
    }

    pub unsafe fn array_row<F: MavrosField>(
        &mut self,
        table: &TableInfo,
        multiplicities: usize,
        i: usize,
        value: F,
    ) {
        unsafe {
            for repetition in 0..self.logup.repetitions {
                let (witnesses, constraints) = self.table_offsets(table, repetition);
                let x = self.w::<F>(witnesses + 2 * i);
                let y = self.w::<F>(witnesses + 2 * i + 1);
                self.row(constraints + 2 * i, self.beta::<F>(repetition), value, -x);
                self.row(
                    constraints + 2 * i + 1,
                    y,
                    self.alpha::<F>(repetition) - F::from(i as u64) * self.w::<F>(0) - x,
                    self.w::<F>(multiplicities + i),
                );
            }
        }
    }

    pub unsafe fn array_sum<F: MavrosField>(&mut self, table: &TableInfo) {
        unsafe {
            for repetition in 0..self.logup.repetitions {
                let (witnesses, constraints) = self.table_offsets(table, repetition);
                let sum = (0..table.length)
                    .map(|i| self.w::<F>(witnesses + 2 * i + 1))
                    .fold(F::ZERO, |sum, value| sum + value);
                self.row(constraints + 2 * table.length, sum, self.w::<F>(0), F::ZERO);
            }
        }
    }

    pub unsafe fn lookup<F: MavrosField>(
        &mut self,
        table: &TableInfo,
        key: F,
        value: Option<F>,
        flag: F,
    ) {
        let rows = if value.is_some() { 2 } else { 1 };
        let width = if table.kind == TableKind::Array { 2 } else { 1 };
        unsafe {
            for repetition in 0..self.logup.repetitions {
                let mut witness = self.lookup_witness + repetition * self.logup.witness_lookups;
                let mut constraint =
                    self.lookup_constraint + repetition * self.logup.constraint_lookups;
                let mut x = F::ZERO;
                if let Some(value) = value {
                    x = self.w::<F>(witness);
                    self.row(constraint, self.beta::<F>(repetition), value, -x);
                    witness += 1;
                    constraint += 1;
                }
                let y = self.w::<F>(witness);
                self.row(constraint, y, self.alpha::<F>(repetition) - x - key, flag);
                let (_, table_constraints) = self.table_offsets(table, repetition);
                *self.outputs[2].slot::<F>() +=
                    self.r::<F>(table_constraints + width * table.length) * y;
            }
        }
        self.lookup_witness += rows;
        self.lookup_constraint += rows;
    }
}
