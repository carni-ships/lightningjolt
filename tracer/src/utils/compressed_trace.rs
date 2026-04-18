//! Compressed trace format for reduced memory bandwidth and storage.
//!
//! This module provides alternative trace representations that use:
//! - Delta encoding for consecutive register values
//! - Run-length encoding for repeated patterns
//! - Smaller integer types where applicable

use serde::{Deserialize, Serialize};

/// A compressed representation of a register value change.
/// Instead of storing the full new value, we store the delta from the previous value.
#[derive(Default, Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct DeltaRegister {
    /// Delta from the previous value. Encoded as i64 for signed deltas.
    pub delta: i64,
    /// Flag indicating if this is a significant change (> 32 bits of non-zero delta)
    pub is_large: bool,
}

impl DeltaRegister {
    pub fn new(previous: u64, current: u64) -> Self {
        let delta = current as i64 - previous as i64;
        let is_large = delta.unsigned_abs() > u32::MAX as u64;
        Self { delta, is_large }
    }

    pub fn expand(&self, previous: u64) -> u64 {
        if self.is_large {
            // For large changes, store the high bits of the current value
            // The delta encodes the upper 32 bits
            ((self.delta as u64) << 32) | (previous & 0xFFFFFFFF)
        } else {
            previous.wrapping_add(self.delta as u64)
        }
    }
}

/// A compressed register state that uses delta encoding.
#[derive(Default, Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CompressedRegisterState {
    /// Delta-encoded destination register old value
    pub rd_old: DeltaRegister,
    /// Delta-encoded destination register new value
    pub rd_new: DeltaRegister,
    /// Delta-encoded source register 1 value
    pub rs1: DeltaRegister,
    /// Delta-encoded source register 2 value
    pub rs2: DeltaRegister,
}

impl CompressedRegisterState {
    pub fn new(previous: &ExpandedRegisterState) -> Self {
        Self {
            rd_old: DeltaRegister::new(0, previous.rd_old),
            rd_new: DeltaRegister::new(0, previous.rd_new),
            rs1: DeltaRegister::new(0, previous.rs1),
            rs2: DeltaRegister::new(0, previous.rs2),
        }
    }

    pub fn update(&self, previous: &ExpandedRegisterState) -> Self {
        Self {
            rd_old: DeltaRegister::new(previous.rd_old, self.rd_old.expand(previous.rd_old)),
            rd_new: DeltaRegister::new(previous.rd_new, self.rd_new.expand(previous.rd_new)),
            rs1: DeltaRegister::new(previous.rs1, self.rs1.expand(previous.rs1)),
            rs2: DeltaRegister::new(previous.rs2, self.rs2.expand(previous.rs2)),
        }
    }
}

/// An expanded (full) register state for decompression.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct ExpandedRegisterState {
    pub rd_old: u64,
    pub rd_new: u64,
    pub rs1: u64,
    pub rs2: u64,
}

/// Run-length encoded segment for repeated patterns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RLESegment<T> {
    /// The value that repeats
    pub value: T,
    /// Number of repetitions (count - 1, since count=1 means just one element)
    pub run_length: u32,
}

impl<T: Clone + PartialEq> RLESegment<T> {
    pub fn new(value: T, count: usize) -> Self {
        Self {
            value,
            run_length: (count - 1) as u32,
        }
    }

    pub fn count(&self) -> usize {
        (self.run_length + 1) as usize
    }
}

/// Compressed trace element using delta encoding.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompressedCycle {
    /// Delta-encoded register state
    pub registers: CompressedRegisterState,
    /// RAM access encoded with smaller address type (only lower 32 bits needed for most accesses)
    pub ram_address_delta: i32,
    /// RAM value delta (post - pre)
    pub ram_value_delta: i64,
    /// Flag indicating if RAM was accessed
    pub ram_was_accessed: bool,
}

/// Statistics about trace compression effectiveness.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub total_cycles: usize,
    pub compressed_size_bytes: usize,
    pub estimated_original_size: usize,
    pub compression_ratio: f64,
    pub large_delta_count: usize,
    pub zero_delta_count: usize,
}

impl CompressionStats {
    pub fn new(total_cycles: usize, compressed_size: usize) -> Self {
        // Original size estimate: 96 bytes per cycle (conservative)
        let original_size = total_cycles * 96;
        let ratio = if original_size > 0 {
            compressed_size as f64 / original_size as f64
        } else {
            1.0
        };
        Self {
            total_cycles,
            compressed_size_bytes: compressed_size,
            estimated_original_size: original_size,
            compression_ratio: ratio,
            large_delta_count: 0,
            zero_delta_count: 0,
        }
    }
}

/// Configuration for trace compression.
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Enable delta encoding for register values
    pub enable_delta_encoding: bool,
    /// Enable run-length encoding for repeated patterns
    pub enable_rle: bool,
    /// Minimum run length to trigger RLE (2 or more)
    pub min_rle_length: usize,
    /// Enable small-address compression (32-bit addresses)
    pub enable_small_addresses: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enable_delta_encoding: true,
            enable_rle: true,
            min_rle_length: 2,
            enable_small_addresses: true,
        }
    }
}

/// An iterator that compresses a trace on-the-fly.
pub struct CompressingTraceIterator<I> {
    inner: I,
    previous_registers: ExpandedRegisterState,
    previous_ram_address: u64,
    previous_ram_value: u64,
    config: CompressionConfig,
}

impl<I: Iterator<Item = crate::instruction::Cycle>> CompressingTraceIterator<I> {
    pub fn new(inner: I, config: CompressionConfig) -> Self {
        Self {
            inner,
            previous_registers: ExpandedRegisterState::default(),
            previous_ram_address: 0,
            previous_ram_value: 0,
            config,
        }
    }
}

impl<I: Iterator<Item = crate::instruction::Cycle>> Iterator for CompressingTraceIterator<I> {
    type Item = CompressedCycle;

    fn next(&mut self) -> Option<Self::Item> {
        let cycle = self.inner.next()?;
        let (ram_access, register_state) = extract_cycle_data(&cycle);

        let registers = if self.config.enable_delta_encoding {
            CompressedRegisterState {
                rd_old: DeltaRegister::new(self.previous_registers.rd_old, register_state.rd_old),
                rd_new: DeltaRegister::new(self.previous_registers.rd_new, register_state.rd_new),
                rs1: DeltaRegister::new(self.previous_registers.rs1, register_state.rs1),
                rs2: DeltaRegister::new(self.previous_registers.rs2, register_state.rs2),
            }
        } else {
            CompressedRegisterState {
                rd_old: DeltaRegister::new(0, register_state.rd_old),
                rd_new: DeltaRegister::new(0, register_state.rd_new),
                rs1: DeltaRegister::new(0, register_state.rs1),
                rs2: DeltaRegister::new(0, register_state.rs2),
            }
        };

        let (ram_address_delta, ram_value_delta, ram_was_accessed) = if ram_access.is_write() {
            let addr = ram_access.address();
            let post_value = ram_access.post_value();
            (
                (addr as i32).wrapping_sub(self.previous_ram_address as i32),
                post_value as i64 - self.previous_ram_value as i64,
                true,
            )
        } else {
            (0, 0, false)
        };

        // Update previous values
        self.previous_registers = register_state;
        if ram_was_accessed {
            self.previous_ram_address = ram_access.address();
            self.previous_ram_value = ram_access.post_value();
        }

        Some(CompressedCycle {
            registers,
            ram_address_delta,
            ram_value_delta,
            ram_was_accessed,
        })
    }
}

/// Extract register and RAM data from a Cycle.
fn extract_cycle_data(cycle: &crate::instruction::Cycle) -> (RAMAccessData, ExpandedRegisterState) {
    let register_state = ExpandedRegisterState {
        rd_old: cycle.rd_write().map(|(_, pre, _)| pre).unwrap_or(0),
        rd_new: cycle.rd_write().map(|(_, _, post)| post).unwrap_or(0),
        rs1: cycle.rs1_read().map(|(_, v)| v).unwrap_or(0),
        rs2: cycle.rs2_read().map(|(_, v)| v).unwrap_or(0),
    };

    let ram_access = match cycle.ram_access() {
        crate::instruction::RAMAccess::Read(r) => RAMAccessData::Read {
            address: r.address,
            pre_value: r.value,
        },
        crate::instruction::RAMAccess::Write(w) => RAMAccessData::Write {
            address: w.address,
            pre_value: w.pre_value,
            post_value: w.post_value,
        },
        crate::instruction::RAMAccess::NoOp => RAMAccessData::NoOp,
    };

    (ram_access, register_state)
}

/// Internal RAM access representation for compression.
#[derive(Debug, Clone, Copy, PartialEq)]
enum RAMAccessData {
    Read { address: u64, pre_value: u64 },
    Write { address: u64, pre_value: u64, post_value: u64 },
    NoOp,
}

impl RAMAccessData {
    fn is_write(&self) -> bool {
        matches!(self, RAMAccessData::Write { .. })
    }

    fn address(&self) -> u64 {
        match self {
            RAMAccessData::Read { address, .. } => *address,
            RAMAccessData::Write { address, .. } => *address,
            RAMAccessData::NoOp => 0,
        }
    }

    fn post_value(&self) -> u64 {
        match self {
            RAMAccessData::Read { pre_value, .. } => *pre_value,
            RAMAccessData::Write { post_value, .. } => *post_value,
            RAMAccessData::NoOp => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_register_small() {
        let delta = DeltaRegister::new(100, 105);
        assert!(!delta.is_large);
        assert_eq!(delta.delta, 5);
        assert_eq!(delta.expand(100), 105);
    }

    #[test]
    fn test_delta_register_large() {
        let delta = DeltaRegister::new(0, 0x1_0000_0000);
        assert!(delta.is_large);
        assert_eq!(delta.expand(0), 0x1_0000_0000);
    }

    #[test]
    fn test_delta_register_negative() {
        let delta = DeltaRegister::new(100, 50);
        assert!(!delta.is_large);
        assert_eq!(delta.delta, -50);
        assert_eq!(delta.expand(100), 50);
    }

    #[test]
    fn test_rle_segment() {
        let seg = RLESegment::new(42u32, 5);
        assert_eq!(seg.count(), 5);
        assert_eq!(seg.run_length, 4);
    }
}
