//! Multi-Tower Trace Adapter for JoltCircleBinius
//!
//! This module maps Jolt's existing Plonkish trace columns into binary tower
//! subfield columns appropriate for each data type.
//!
//! # Overview
//!
//! Jolt currently uses one monolithic BLS12-381 field for all trace columns.
//! Binary tower fields allow each column to use the smallest field that fits
//! its data - reducing field element size and enabling bit-packing.
//!
//! # Column Mapping
//!
//! | Jolt Trace Column | Data Width | Tower Subfield | Bit-Packing |
//! |-------------------|------------|----------------|-------------|
//! | Instruction opcode | 8-bit | F_{2^8} | 16 opcodes per element |
//! | Register values | 32-bit | F_{2^{32}} | Native |
//! | Memory addresses | 64-bit | F_{2^{64}} | Native |
//! | Lookup table indices | 8-bit | F_{2^8} | 16 indices per element |
//! | FRI challenges | 128-bit | F_{2^{128}} | Native |
//!
//! # Usage
//!
//! ```rust
//! use jolt_core::poly::commitment::circle_binius::{TowerTrace, BinaryField32b};
//!
//! // Convert Jolt trace to tower format
//! let tower_trace = TowerTrace::from_jolt_trace(&jolt_trace);
//!
//! // Get evaluations for FFT
//! let evals = tower_trace.to_evaluations();
//! ```

use super::fields::{BinaryField128b, BinaryField32b, BinaryField64b, BinaryField8b};
use num_traits::Zero;

/// Number of trace rows after padding to power of 2
const fn next_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1;
    while p < n {
        p *= 2;
    }
    p
}

/// A trace column containing values of type F
#[derive(Debug, Clone)]
pub struct Column<F> {
    /// Evaluations at trace points (length = num_rows)
    pub evaluations: Vec<F>,
    /// Original length before padding
    pub original_len: usize,
}

impl<F: Clone> Column<F> {
    /// Create a new column from evaluations
    pub fn new(evaluations: Vec<F>, original_len: usize) -> Self {
        Self {
            evaluations,
            original_len,
        }
    }

    /// Get the number of rows (after padding)
    pub fn num_rows(&self) -> usize {
        self.evaluations.len()
    }

    /// Get original length before padding
    pub fn original_len(&self) -> usize {
        self.original_len
    }
}

/// Tower-formatted trace with columns split by data width
///
/// This structure organizes Jolt's trace columns into binary tower subfields,
/// allowing efficient FFT operations and bit-packing.
#[derive(Debug, Clone)]
pub struct TowerTrace {
    /// 8-bit columns: opcodes, lookup indices
    pub byte_columns: Vec<Column<BinaryField8b>>,
    /// 32-bit columns: registers, memory words
    pub word_columns: Vec<Column<BinaryField32b>>,
    /// 64-bit columns: addresses
    pub addr_columns: Vec<Column<BinaryField64b>>,
    /// 128-bit columns: FRI challenges, security params
    pub security_columns: Vec<Column<BinaryField128b>>,
    /// Total number of trace rows (padded to power of 2)
    pub num_rows: usize,
    /// Log of num_rows (for FFT planning)
    pub log_num_rows: usize,
}

impl TowerTrace {
    /// Create a new empty tower trace
    pub fn new() -> Self {
        Self {
            byte_columns: Vec::new(),
            word_columns: Vec::new(),
            addr_columns: Vec::new(),
            security_columns: Vec::new(),
            num_rows: 0,
            log_num_rows: 0,
        }
    }

    /// Get total number of trace rows (padded to power of 2)
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Get log of number of rows (for FFT)
    pub fn log_num_rows(&self) -> usize {
        self.log_num_rows
    }

    /// Get total number of columns
    pub fn num_columns(&self) -> usize {
        self.byte_columns.len() + self.word_columns.len() + self.addr_columns.len() + self.security_columns.len()
    }

    /// Add an 8-bit column (opcode, index, etc.)
    pub fn add_byte_column(&mut self, mut evaluations: Vec<BinaryField8b>) {
        let original_len = evaluations.len();
        let padded_len = next_power_of_2(evaluations.len());
        if padded_len > self.num_rows {
            self.num_rows = padded_len;
            self.log_num_rows = (padded_len as f64).log2() as usize;
        }
        // Pad to num_rows
        evaluations.resize(self.num_rows, BinaryField8b::zero());
        self.byte_columns.push(Column::new(evaluations, original_len));
    }

    /// Add a 32-bit column (register, memory word)
    pub fn add_word_column(&mut self, mut evaluations: Vec<BinaryField32b>) {
        let original_len = evaluations.len();
        let padded_len = next_power_of_2(evaluations.len());
        if padded_len > self.num_rows {
            self.num_rows = padded_len;
            self.log_num_rows = (padded_len as f64).log2() as usize;
        }
        evaluations.resize(self.num_rows, BinaryField32b::zero());
        self.word_columns.push(Column::new(evaluations, original_len));
    }

    /// Add a 64-bit column (address)
    pub fn add_addr_column(&mut self, mut evaluations: Vec<BinaryField64b>) {
        let original_len = evaluations.len();
        let padded_len = next_power_of_2(evaluations.len());
        if padded_len > self.num_rows {
            self.num_rows = padded_len;
            self.log_num_rows = (padded_len as f64).log2() as usize;
        }
        evaluations.resize(self.num_rows, BinaryField64b::zero());
        self.addr_columns.push(Column::new(evaluations, original_len));
    }

    /// Add a 128-bit column (security param, hash output)
    pub fn add_security_column(&mut self, mut evaluations: Vec<BinaryField128b>) {
        let original_len = evaluations.len();
        let padded_len = next_power_of_2(evaluations.len());
        if padded_len > self.num_rows {
            self.num_rows = padded_len;
            self.log_num_rows = (padded_len as f64).log2() as usize;
        }
        evaluations.resize(self.num_rows, BinaryField128b::zero());
        self.security_columns.push(Column::new(evaluations, original_len));
    }

    /// Convert to flat evaluations for FFT
    ///
    /// Returns a single vector of BinaryField32b elements (the "native" field
    /// for Jolt operations) with all columns packed and padded.
    pub fn to_evaluations(&self) -> Vec<BinaryField32b> {
        let mut result = Vec::with_capacity(self.num_rows * self.num_columns());

        // Interleave columns row by row for FFT-friendly layout
        for row in 0..self.num_rows {
            // Add 8-bit columns (convert to 32-bit)
            for col in &self.byte_columns {
                let val: u8 = col.evaluations[row].to_repr();
                result.push(BinaryField32b::from_repr(val as u32));
            }
            // Add 32-bit columns
            for col in &self.word_columns {
                result.push(col.evaluations[row]);
            }
            // Add 64-bit columns (as two 32-bit halves)
            for col in &self.addr_columns {
                let val = col.evaluations[row].to_repr();
                result.push(BinaryField32b::from_repr(val as u32));
                result.push(BinaryField32b::from_repr((val >> 32) as u32));
            }
            // Add 128-bit columns (as four 32-bit quarters)
            for col in &self.security_columns {
                let (lo, hi) = col.evaluations[row].to_repr();
                result.push(BinaryField32b::from_repr(lo as u32));
                result.push(BinaryField32b::from_repr((lo >> 32) as u32));
                result.push(BinaryField32b::from_repr(hi as u32));
                result.push(BinaryField32b::from_repr((hi >> 32) as u32));
            }
        }

        result
    }

    /// Get the total number of evaluations in flat form
    pub fn total_evaluations(&self) -> usize {
        self.num_rows * self.num_columns()
    }

    /// Create from flat evaluations (inverse of to_evaluations)
    pub fn from_evaluations(
        evaluations: &[BinaryField32b],
        num_rows: usize,
        num_byte_cols: usize,
        num_word_cols: usize,
        num_addr_cols: usize,
        num_security_cols: usize,
    ) -> Self {
        let mut tower = Self::new();
        tower.num_rows = num_rows;
        tower.log_num_rows = (num_rows as f64).log2() as usize;

        // evals layout: row0 has all columns interleaved, then row1, etc.
        // For each row: byte_cols[0..num_byte_cols], word_cols[0..num_word_cols], addr_cols[0..num_addr_cols]*2, security_cols[0..num_security_cols]*4

        // Helper to calculate index in evals for a given row and column offset
        let mut byte_idx = 0;
        let mut word_idx = num_byte_cols;
        let mut addr_idx = num_byte_cols + num_word_cols;
        let mut security_idx = addr_idx + num_addr_cols * 2;

        // 8-bit columns (convert from 32-bit)
        for col in 0..num_byte_cols {
            let mut evals = Vec::with_capacity(num_rows);
            for row in 0..num_rows {
                let idx = row * (num_byte_cols + num_word_cols + num_addr_cols * 2 + num_security_cols * 4) + col;
                let val: u32 = evaluations[idx].to_repr();
                evals.push(BinaryField8b::from_repr(val as u8));
            }
            let original_len = evals.iter().rposition(|e| e.is_zero()).map(|i| i + 1).unwrap_or(0);
            tower.byte_columns.push(Column::new(evals, original_len));
        }

        // 32-bit columns
        for col in 0..num_word_cols {
            let mut evals = Vec::with_capacity(num_rows);
            let col_offset = num_byte_cols + col;
            for row in 0..num_rows {
                let idx = row * (num_byte_cols + num_word_cols + num_addr_cols * 2 + num_security_cols * 4) + col_offset;
                evals.push(evaluations[idx]);
            }
            let original_len = evals.iter().rposition(|e| e.is_zero()).map(|i| i + 1).unwrap_or(0);
            tower.word_columns.push(Column::new(evals, original_len));
        }

        // 64-bit columns (2 u32s per value)
        for col in 0..num_addr_cols {
            let mut evals = Vec::with_capacity(num_rows);
            let col_offset = num_byte_cols + num_word_cols + col * 2;
            for row in 0..num_rows {
                let idx = row * (num_byte_cols + num_word_cols + num_addr_cols * 2 + num_security_cols * 4) + col_offset;
                let lo: u32 = evaluations[idx].to_repr();
                let hi: u32 = evaluations[idx + 1].to_repr();
                let combined = (lo as u64) | ((hi as u64) << 32);
                evals.push(BinaryField64b::from_repr(combined));
            }
            let original_len = evals.iter().rposition(|e| e.is_zero()).map(|i| i + 1).unwrap_or(0);
            tower.addr_columns.push(Column::new(evals, original_len));
        }

        // 128-bit columns (4 u32s per value)
        for col in 0..num_security_cols {
            let mut evals = Vec::with_capacity(num_rows);
            let col_offset = num_byte_cols + num_word_cols + num_addr_cols * 2 + col * 4;
            for row in 0..num_rows {
                let idx = row * (num_byte_cols + num_word_cols + num_addr_cols * 2 + num_security_cols * 4) + col_offset;
                let lo0: u32 = evaluations[idx].to_repr();
                let lo1: u32 = evaluations[idx + 1].to_repr();
                let hi0: u32 = evaluations[idx + 2].to_repr();
                let hi1: u32 = evaluations[idx + 3].to_repr();
                let lo = (lo0 as u64) | ((lo1 as u64) << 32);
                let hi = (hi0 as u64) | ((hi1 as u64) << 32);
                evals.push(BinaryField128b::from_repr((lo, hi)));
            }
            let original_len = evals.iter().rposition(|e| e.is_zero()).map(|i| i + 1).unwrap_or(0);
            tower.security_columns.push(Column::new(evals, original_len));
        }

        tower
    }
}

impl Default for TowerTrace {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tower_trace() {
        let tower = TowerTrace::new();
        assert_eq!(tower.num_rows(), 0);
        assert_eq!(tower.num_columns(), 0);
    }

    #[test]
    fn test_add_byte_column() {
        let mut tower = TowerTrace::new();
        tower.add_byte_column(vec![
            BinaryField8b(1),
            BinaryField8b(2),
            BinaryField8b(3),
        ]);
        assert_eq!(tower.num_rows(), next_power_of_2(3));
        assert_eq!(tower.byte_columns.len(), 1);
        assert_eq!(tower.byte_columns[0].original_len(), 3);
    }

    #[test]
    fn test_round_trip() {
        let mut tower = TowerTrace::new();

        // Add some columns
        tower.add_byte_column(vec![BinaryField8b(1), BinaryField8b(2), BinaryField8b(3), BinaryField8b(4)]);
        tower.add_word_column(vec![BinaryField32b(10), BinaryField32b(20), BinaryField32b(30), BinaryField32b(40)]);

        let evals = tower.to_evaluations();

        // Reconstruct
        let reconstructed = TowerTrace::from_evaluations(
            &evals,
            tower.num_rows(),
            1, // num_byte_cols
            1, // num_word_cols
            0, // num_addr_cols
            0, // num_security_cols
        );

        assert_eq!(reconstructed.num_rows(), tower.num_rows());
        assert_eq!(reconstructed.byte_columns.len(), 1);
        assert_eq!(reconstructed.word_columns.len(), 1);
        assert_eq!(reconstructed.byte_columns[0].evaluations[0], tower.byte_columns[0].evaluations[0]);
        assert_eq!(reconstructed.word_columns[0].evaluations[0], tower.word_columns[0].evaluations[0]);
    }

    #[test]
    fn test_packing() {
        let mut tower = TowerTrace::new();

        // Add columns of different sizes
        tower.add_byte_column(vec![BinaryField8b(1); 4]);
        tower.add_word_column(vec![BinaryField32b(1); 4]);

        let evals = tower.to_evaluations();
        // Should have 4 rows x (1 + 1) = 8 columns
        assert_eq!(evals.len(), 8);

        // Verify packing: byte, word, byte, word, ...
        assert_eq!(evals[0], BinaryField32b(1)); // byte col 0, row 0
        assert_eq!(evals[1], BinaryField32b(1)); // word col 0, row 0
        assert_eq!(evals[2], BinaryField32b(1)); // byte col 0, row 1
        assert_eq!(evals[3], BinaryField32b(1)); // word col 0, row 1
    }
}