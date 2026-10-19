//! Forward evaluation of the constraint program at an arbitrary witness vector.
//!
//! Witness references are field values in this interpretation. No AD nodes or
//! matrix rows are constructed. Lookup rows use the same layout as the R1CS.

use mavros_artifacts::MavrosField;

use crate::{
    ConstraintsLayout, TableKind, WitnessLayout,
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
    // FIELD-ASSUMPTION: L4-logup-challenges
    // One LogUp challenge pair (alpha, beta); no K-challenge replication.
    challenge: usize,
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
        values: impl Iterator<Item = F>,
    ) {
        unsafe {
            let mut sum = F::ZERO;
            let alpha = self.w::<F>(self.challenge);
            let w0 = self.w::<F>(0);
            // Range tables need no beta column.
            let beta = if table.kind == TableKind::RangeCheck {
                F::ZERO
            } else {
                self.w::<F>(self.challenge + 1)
            };
            for (i, value) in values.enumerate() {
                let y = self.w::<F>(table.elem_inverses_witness_section_offset + i);
                self.row(
                    table.elem_inverses_constraint_section_offset + i,
                    y,
                    alpha - F::from(i as u64) * w0 + beta * value,
                    self.w::<F>(multiplicities + i),
                );
                sum += y;
            }
            self.row(
                table.elem_inverses_constraint_section_offset + table.length,
                sum,
                w0,
                F::ZERO,
            );
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
            let x = self.w::<F>(table.elem_inverses_witness_section_offset + 2 * i);
            let y = self.w::<F>(table.elem_inverses_witness_section_offset + 2 * i + 1);
            self.row(
                table.elem_inverses_constraint_section_offset + 2 * i,
                self.w::<F>(self.challenge + 1),
                value,
                -x,
            );
            self.row(
                table.elem_inverses_constraint_section_offset + 2 * i + 1,
                y,
                self.w::<F>(self.challenge) - F::from(i as u64) * self.w::<F>(0) - x,
                self.w::<F>(multiplicities + i),
            );
        }
    }

    pub unsafe fn array_sum<F: MavrosField>(&mut self, table: &TableInfo) {
        unsafe {
            let sum = (0..table.length)
                .map(|i| self.w::<F>(table.elem_inverses_witness_section_offset + 2 * i + 1))
                .fold(F::ZERO, |sum, value| sum + value);
            self.row(
                table.elem_inverses_constraint_section_offset + 2 * table.length,
                sum,
                self.w::<F>(0),
                F::ZERO,
            );
        }
    }

    pub unsafe fn lookup<F: MavrosField>(
        &mut self,
        table: &TableInfo,
        key: F,
        value: Option<F>,
        flag: F,
    ) {
        unsafe {
            let mut x = F::ZERO;
            if let Some(value) = value {
                x = self.w::<F>(self.lookup_witness);
                self.row(
                    self.lookup_constraint,
                    self.w::<F>(self.challenge + 1),
                    value,
                    -x,
                );
                self.lookup_witness += 1;
                self.lookup_constraint += 1;
            }
            let y = self.w::<F>(self.lookup_witness);
            self.row(
                self.lookup_constraint,
                y,
                self.w::<F>(self.challenge) - x - key,
                flag,
            );
            self.lookup_witness += 1;
            self.lookup_constraint += 1;
            let width = if table.kind == TableKind::Array { 2 } else { 1 };
            *self.outputs[2].slot::<F>() += self
                .r::<F>(table.elem_inverses_constraint_section_offset + width * table.length)
                * y;
        }
    }
}
