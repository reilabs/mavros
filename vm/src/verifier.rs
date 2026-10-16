//! Forward evaluation of the constraint program at an arbitrary witness vector.
//!
//! Witness references are field values in this interpretation. No AD nodes or
//! matrix rows are constructed. Lookup rows use the same layout as the R1CS.

use ark_ff::AdditiveGroup;

use crate::{ConstraintsLayout, Field, TableKind, WitnessLayout, bytecode::TableInfo};

/// Borrowed execution buffers, installed only after the runner checks their lengths.
/// All pointers remain valid until dispatch finishes. Only verifier opcodes may
/// access this member of the VM's execution-data union.
#[derive(Clone, Copy)]
pub(crate) struct VerifierArrays {
    pub outputs: [*mut Field; 3],
    pub witness: *const Field,
    pub coeffs: *const Field,
    pub witness_cursor: usize,
    pub constraint_cursor: usize,
    challenge: usize,
    multiplicities: usize,
    table_witness: usize,
    table_constraint: usize,
    lookup_witness: usize,
    lookup_constraint: usize,
}

impl VerifierArrays {
    pub fn new(
        outputs: [*mut Field; 3],
        witness: *const Field,
        coeffs: *const Field,
        wl: WitnessLayout,
        cl: ConstraintsLayout,
    ) -> Self {
        Self {
            outputs,
            witness,
            coeffs,
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

    pub unsafe fn w(&self, index: usize) -> Field {
        unsafe { *self.witness.add(index) }
    }
    unsafe fn r(&self, index: usize) -> Field {
        unsafe { *self.coeffs.add(index) }
    }

    /// Accumulate one row's three forward-evaluated linear combinations.
    unsafe fn row(&mut self, index: usize, a: Field, b: Field, c: Field) {
        unsafe {
            let r = self.r(index);
            for (output, value) in self.outputs.iter().zip([a, b, c]) {
                **output += r * value;
            }
        }
    }

    /// Reserve a table in the deterministic order used by matrix generation.
    pub unsafe fn table(&mut self, kind: TableKind, length: usize) -> (TableInfo, usize) {
        let width = if kind == TableKind::Array { 2 } else { 1 };
        let table = TableInfo {
            multiplicities_wit: std::ptr::null_mut(),
            num_indices: 1,
            kind,
            length,
            elem_inverses_witness_section_offset: self.table_witness,
            elem_inverses_constraint_section_offset: self.table_constraint,
        };
        let multiplicities = self.multiplicities;
        self.multiplicities += length;
        self.table_witness += width * length;
        self.table_constraint += width * length + 1;
        (table, multiplicities)
    }

    /// Range, spread and pow2 tables have constant entries and one folded row per entry.
    pub unsafe fn constant_table(
        &mut self,
        table: &TableInfo,
        multiplicities: usize,
        values: impl Iterator<Item = Field>,
    ) {
        unsafe {
            let mut sum = Field::ZERO;
            let alpha = self.w(self.challenge);
            let w0 = self.w(0);
            // Range tables need no beta column.
            let beta = if table.kind == TableKind::RangeCheck {
                Field::ZERO
            } else {
                self.w(self.challenge + 1)
            };
            for (i, value) in values.enumerate() {
                let y = self.w(table.elem_inverses_witness_section_offset + i);
                self.row(
                    table.elem_inverses_constraint_section_offset + i,
                    y,
                    alpha - Field::from(i as u64) * w0 + beta * value,
                    self.w(multiplicities + i),
                );
                sum += y;
            }
            self.row(
                table.elem_inverses_constraint_section_offset + table.length,
                sum,
                w0,
                Field::ZERO,
            );
        }
    }

    pub unsafe fn array_row(
        &mut self,
        table: &TableInfo,
        multiplicities: usize,
        i: usize,
        value: Field,
    ) {
        unsafe {
            let x = self.w(table.elem_inverses_witness_section_offset + 2 * i);
            let y = self.w(table.elem_inverses_witness_section_offset + 2 * i + 1);
            self.row(
                table.elem_inverses_constraint_section_offset + 2 * i,
                self.w(self.challenge + 1),
                value,
                -x,
            );
            self.row(
                table.elem_inverses_constraint_section_offset + 2 * i + 1,
                y,
                self.w(self.challenge) - Field::from(i as u64) * self.w(0) - x,
                self.w(multiplicities + i),
            );
        }
    }

    pub unsafe fn array_sum(&mut self, table: &TableInfo) {
        unsafe {
            let sum = (0..table.length)
                .map(|i| self.w(table.elem_inverses_witness_section_offset + 2 * i + 1))
                .sum();
            self.row(
                table.elem_inverses_constraint_section_offset + 2 * table.length,
                sum,
                self.w(0),
                Field::ZERO,
            );
        }
    }

    pub unsafe fn lookup(
        &mut self,
        table: &TableInfo,
        key: Field,
        value: Option<Field>,
        flag: Field,
    ) {
        unsafe {
            let mut x = Field::ZERO;
            if let Some(value) = value {
                x = self.w(self.lookup_witness);
                self.row(
                    self.lookup_constraint,
                    self.w(self.challenge + 1),
                    value,
                    -x,
                );
                self.lookup_witness += 1;
                self.lookup_constraint += 1;
            }
            let y = self.w(self.lookup_witness);
            self.row(
                self.lookup_constraint,
                y,
                self.w(self.challenge) - x - key,
                flag,
            );
            self.lookup_witness += 1;
            self.lookup_constraint += 1;
            let width = if table.kind == TableKind::Array { 2 } else { 1 };
            *self.outputs[2] +=
                self.r(table.elem_inverses_constraint_section_offset + width * table.length) * y;
        }
    }
}
