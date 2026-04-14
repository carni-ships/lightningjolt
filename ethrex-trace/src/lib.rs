//! Ethrex Trace Generator - Type B1 zkEVM Trace Generation
//!
//! This crate wraps ethrex's EVM execution with step-by-step trace generation
//! for use with Jolt's Type B1 zkEVM verification.
//!
//! ## Architecture
//!
//! Host (this crate):
//! - Executes EVM bytecode using ethrex VM
//! - Generates step-by-step traces via StepTracer
//! - Outputs 41-byte per step format for guest verification
//!
//! Guest (jolt-evm-guest):
//! - Verifies trace constraints using check_advice!
//! - Validates opcode gas costs, PC bounds, stack deltas

#![no_std]
extern crate alloc;

use alloc::vec::Vec;

/// 41-byte step trace format:
/// [opcode: u8][pc: u64][gas_before: u64][gas_used: u64][stack_len_before: u64][stack_len_after: u64]
pub const STEP_SIZE: usize = 41;

/// Result of executing bytecode with tracing
pub struct ExecutionResult {
    /// The trace data (41 bytes per step)
    pub trace_data: Vec<u8>,
    /// Number of steps in the trace
    pub steps_len: usize,
    /// Final gas remaining
    pub final_gas: u64,
    /// Whether execution was successful
    pub success: bool,
}

/// Trace generator that collects step-by-step execution traces.
pub struct TraceGenerator {
    traces: Vec<u8>,
}

impl TraceGenerator {
    /// Create a new trace generator.
    pub fn new() -> Self {
        Self { traces: Vec::new() }
    }

    /// Clear the traces buffer.
    pub fn clear(&mut self) {
        self.traces.clear();
    }

    /// Get the raw trace data.
    pub fn trace_data(&self) -> &[u8] {
        &self.traces
    }

    /// Get the number of steps in the trace.
    pub fn steps_len(&self) -> usize {
        self.traces.len() / STEP_SIZE
    }

    /// Add a step to the trace.
    pub fn add_step(&mut self, opcode: u8, pc: u64, gas_before: u64, gas_used: u64, stack_len_before: usize, stack_len_after: usize) {
        self.traces.push(opcode);
        self.traces.extend_from_slice(&pc.to_le_bytes());
        self.traces.extend_from_slice(&gas_before.to_le_bytes());
        self.traces.extend_from_slice(&gas_used.to_le_bytes());
        self.traces.extend_from_slice(&(stack_len_before as u64).to_le_bytes());
        self.traces.extend_from_slice(&(stack_len_after as u64).to_le_bytes());
    }
}

impl Default for TraceGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> ethrex_levm::step_tracer::StepTracer for &'a mut TraceGenerator {
    fn on_step(&mut self, trace: ethrex_levm::step_tracer::StepTrace) {
        self.add_step(
            trace.opcode,
            trace.pc,
            trace.gas_before,
            trace.gas_used,
            trace.stack_len_before,
            trace.stack_len_after,
        );
    }
}
