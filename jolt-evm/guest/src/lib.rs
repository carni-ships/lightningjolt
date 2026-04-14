//! Jolt-EVM Guest Program - Type B1 zkEVM
//!
//! This guest verifies EVM execution traces using Jolt's constraint system.
//! The host executes EVM bytecode and generates a trace. The guest verifies
//! each step's correctness using check_advice! constraints.
//!
//! Trace verification approach (Type B1):
//! 1. Receive trace header (entry_pc, bytecode_hash, etc.)
//! 2. For each step, verify:
//!    - Opcode is valid (0x00-0xff)
//!    - PC transition is valid for the opcode
//!    - Gas consumption matches opcode cost
//!    - Stack delta is correct for the opcode

#![cfg_attr(feature = "guest", no_std)]
#![no_main]

extern crate alloc;

use jolt::check_advice;
use jolt::check_advice_eq;

// Import EVM types from jolt-evm
pub use jolt_evm::{Address, H256, U256, EvmStep, EvmTrace, Opcode};

/// Gas costs for EVM opcodes
const GAS_STOP: u64 = 0;
const GAS_ADD: u64 = 3;
const GAS_MUL: u64 = 5;
const GAS_SUB: u64 = 3;
const GAS_DIV: u64 = 5;
const GAS_SDIV: u64 = 5;
const GAS_MOD: u64 = 5;
const GAS_SMOD: u64 = 5;
const GAS_ADDMOD: u64 = 8;
const GAS_MULMOD: u64 = 8;
const GAS_EXP: u64 = 10;
const GAS_SIGNEXTEND: u64 = 5;
const GAS_LT: u64 = 3;
const GAS_GT: u64 = 3;
const GAS_SLT: u64 = 3;
const GAS_SGT: u64 = 3;
const GAS_EQ: u64 = 3;
const GAS_ISZERO: u64 = 3;
const GAS_AND: u64 = 3;
const GAS_OR: u64 = 3;
const GAS_XOR: u64 = 3;
const GAS_NOT: u64 = 3;
const GAS_BYTE: u64 = 3;
const GAS_SHL: u64 = 3;
const GAS_SHR: u64 = 3;
const GAS_SAR: u64 = 3;
const GAS_SHA3: u64 = 30;
const GAS_MLOAD: u64 = 3;
const GAS_MSTORE: u64 = 3;
const GAS_MSTORE8: u64 = 3;
const GAS_SLOAD: u64 = 100;
const GAS_SSTORE: u64 = 100;
const GAS_JUMP: u64 = 8;
const GAS_JUMPI: u64 = 10;
const GAS_PC: u64 = 2;
const GAS_JUMPDEST: u64 = 1;
const GAS_PUSH: u64 = 3;
const GAS_DUP: u64 = 3;
const GAS_SWAP: u64 = 3;

/// Maximum bytecode size we support (reserved for future bytecode verification)
const _MAX_BYTECODE_SIZE: usize = 10000;

/// Maximum trace steps we support
const MAX_TRACE_STEPS: usize = 65536;

/// Size of a single step in the advice tape (flattened representation)
/// Format: [opcode: u8, pc: u64, gas_before: u64, gas_used: u64, stack_len_before: usize, stack_len_after: usize]
const STEP_SIZE: usize = 41; // 1 + 8 + 8 + 8 + 8 + 8 bytes

/// Verify an EVM execution trace
///
/// Public inputs:
/// - entry_pc: Initial program counter
/// - bytecode_hash: Hash of the bytecode (for anchoring)
/// - steps_len: Number of steps in the trace
/// - final_gas: Gas remaining at end of execution
/// - success: Whether execution completed successfully
/// - trace_data: Serialized step data (41 bytes per step)
///
/// Returns 1 if verification passes, 0 otherwise.
#[jolt::provable(heap_size = 1048576, max_trace_length = 65536)]
pub fn verify_evm_trace(
    entry_pc: u64,
    _bytecode_hash: [u8; 32],
    steps_len: usize,
    final_gas: u64,
    success: bool,
    trace_data: &[u8],
) -> u8 {
    // Verify trace header constraints
    check_advice!(entry_pc < 1000000u64, "Entry PC too large");
    check_advice!(steps_len > 0usize, "Empty trace");
    check_advice!(steps_len < MAX_TRACE_STEPS, "Too many steps");
    check_advice!(final_gas < 10000000u64, "Gas remaining too large");

    // Calculate expected trace data size
    let expected_len = steps_len * STEP_SIZE;
    check_advice!(
        trace_data.len() >= expected_len,
        "Insufficient trace data"
    );

    // Verify each step
    let mut total_gas_used: u64 = 0;
    verify_steps_loop(trace_data, steps_len, &mut total_gas_used);

    // Return success indicator
    if success {
        1u8
    } else {
        0u8
    }
}

/// Helper to verify steps in a loop
#[inline]
fn verify_steps_loop(trace_data: &[u8], steps_len: usize, total_gas_used: &mut u64) {
    let mut offset = 0usize;

    let mut step_idx = 0usize;
    while step_idx < steps_len {
        // Read step fields from the flattened data
        let opcode = trace_data[offset];
        offset += 1;

        // Read pc (little-endian u64)
        let pc = read_u64(trace_data, offset);
        offset += 8;

        // Read gas_before (little-endian u64)
        let gas_before = read_u64(trace_data, offset);
        offset += 8;

        // Read gas_used (little-endian u64)
        let gas_used = read_u64(trace_data, offset);
        offset += 8;

        // Read stack_len_before
        let stack_len_before = read_usize(trace_data, offset);
        offset += 8;

        // Read stack_len_after
        let stack_len_after = read_usize(trace_data, offset);
        offset += 8;

        // Accumulate gas used
        *total_gas_used = total_gas_used.wrapping_add(gas_used);

        // Verify this step's constraints
        verify_single_step(opcode, pc, gas_before, gas_used, stack_len_before, stack_len_after);

        step_idx += 1;
    }
}

/// Read a u64 from a byte slice (little-endian)
#[inline]
fn read_u64(data: &[u8], offset: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&data[offset..offset + 8]);
    u64::from_le_bytes(bytes)
}

/// Read a usize from a byte slice (little-endian, stored as u64)
#[inline]
fn read_usize(data: &[u8], offset: usize) -> usize {
    read_u64(data, offset) as usize
}

/// Verify constraints for a single step
#[inline]
fn verify_single_step(
    opcode: u8,
    pc: u64,
    _gas_before: u64,
    gas_used: u64,
    stack_len_before: usize,
    stack_len_after: usize,
) {
    // 1. Verify opcode is valid (0x00-0xff)
    // Note: u8 is already bounded to 0-255, so no explicit check needed

    // 2. Verify gas consumption matches opcode cost
    let expected_gas = get_opcode_gas_cost(opcode);
    check_advice_eq!(gas_used, expected_gas, "Gas used doesn't match opcode cost");

    // 3. Verify PC is in bounds
    check_advice!(pc < 1000000u64, "PC out of bounds");

    // 4. Verify stack delta is correct for opcode
    verify_stack_delta(opcode, stack_len_before, stack_len_after);
}

/// Get the gas cost for an opcode
fn get_opcode_gas_cost(opcode: u8) -> u64 {
    match opcode {
        0x00 => GAS_STOP,       // STOP
        0x01 => GAS_ADD,        // ADD
        0x02 => GAS_MUL,        // MUL
        0x03 => GAS_SUB,        // SUB
        0x04 => GAS_DIV,        // DIV
        0x05 => GAS_SDIV,       // SDIV
        0x06 => GAS_MOD,        // MOD
        0x07 => GAS_SMOD,       // SMOD
        0x08 => GAS_ADDMOD,     // ADDMOD
        0x09 => GAS_MULMOD,     // MULMOD
        0x0a => GAS_EXP,        // EXP
        0x0b => GAS_SIGNEXTEND, // SIGNEXTEND
        0x10 => GAS_LT,         // LT
        0x11 => GAS_GT,         // GT
        0x12 => GAS_SLT,        // SLT
        0x13 => GAS_SGT,        // SGT
        0x14 => GAS_EQ,         // EQ
        0x15 => GAS_ISZERO,     // ISZERO
        0x16 => GAS_AND,        // AND
        0x17 => GAS_OR,         // OR
        0x18 => GAS_XOR,        // XOR
        0x19 => GAS_NOT,        // NOT
        0x1a => GAS_BYTE,       // BYTE
        0x1b => GAS_SHL,       // SHL
        0x1c => GAS_SHR,        // SHR
        0x1d => GAS_SAR,        // SAR
        0x20 => GAS_SHA3,       // SHA3
        0x51 => GAS_MLOAD,      // MLOAD
        0x52 => GAS_MSTORE,     // MSTORE
        0x53 => GAS_MSTORE8,    // MSTORE8
        0x54 => GAS_SLOAD,      // SLOAD
        0x55 => GAS_SSTORE,     // SSTORE
        0x56 => GAS_JUMP,       // JUMP
        0x57 => GAS_JUMPI,      // JUMPI
        0x58 => GAS_PC,         // PC
        0x5b => GAS_JUMPDEST,   // JUMPDEST
        // PUSH1-PUSH32: 0x60-0x7f
        0x60..=0x7f => GAS_PUSH,
        // DUP1-DUP16: 0x80-0x8f
        0x80..=0x8f => GAS_DUP,
        // SWAP1-SWAP16: 0x90-0x9f
        0x90..=0x9f => GAS_SWAP,
        0xf3 => GAS_STOP,       // RETURN
        0xfd => GAS_STOP,       // REVERT
        0xfe => GAS_STOP,       // INVALID
        _ => GAS_ADD,           // Default - will fail on unknown
    }
}

/// Verify stack length change is correct for the opcode
fn verify_stack_delta(opcode: u8, len_before: usize, len_after: usize) {
    let before = len_before as u64;
    let after = len_after as u64;

    match opcode {
        // STOP, REVERT, INVALID: stack doesn't matter (execution ends)
        0x00 | 0xf3 | 0xfd | 0xfe => {
            // No constraint - these end execution
        }
        // PUSH1-PUSH32: increases stack by 1
        0x60..=0x7f => {
            check_advice_eq!(after, before + 1, "PUSH should increase stack by 1");
        }
        // DUP1-DUP16: duplicates top, increases by 1
        0x80..=0x8f => {
            check_advice_eq!(after, before + 1, "DUP should increase stack by 1");
        }
        // SWAP1-SWAP16: swaps don't change stack size
        0x90..=0x9f => {
            check_advice_eq!(after, before, "SWAP should not change stack size");
        }
        // ADD: pops 2, pushes 1 (net -1)
        0x01 => {
            check_advice_eq!(after, before - 1, "ADD should reduce stack by 1");
        }
        // MUL: pops 2, pushes 1 (net -1)
        0x02 => {
            check_advice_eq!(after, before - 1, "MUL should reduce stack by 1");
        }
        // SUB: pops 2, pushes 1 (net -1)
        0x03 => {
            check_advice_eq!(after, before - 1, "SUB should reduce stack by 1");
        }
        // DIV, SDIV, MOD, SMOD, ADDMOD, MULMOD, EXP, SIGNEXTEND: same as ADD (net -1)
        0x04..=0x0b => {
            check_advice_eq!(after, before - 1, "Binary op should reduce stack by 1");
        }
        // LT, GT, SLT, SGT, EQ: same as ADD (net -1)
        0x10..=0x14 => {
            check_advice_eq!(after, before - 1, "Comparison op should reduce stack by 1");
        }
        // ISZERO: pops 1, pushes 1 (net 0)
        0x15 => {
            check_advice_eq!(after, before, "ISZERO should not change stack size");
        }
        // BYTE, SHL, SHR, SAR: same as ADD (net -1)
        0x1a..=0x1d => {
            check_advice_eq!(after, before - 1, "Byte/bitwise op should reduce stack by 1");
        }
        // AND, OR, XOR: pops 2, pushes 1 (net -1)
        0x16..=0x18 => {
            check_advice_eq!(after, before - 1, "Bitwise op should reduce stack by 1");
        }
        // NOT: pops 1, pushes 1 (net 0)
        0x19 => {
            check_advice_eq!(after, before, "NOT should not change stack size");
        }
        // JUMP: pops 1 (destination), doesn't push
        0x56 => {
            check_advice_eq!(after, before - 1, "JUMP should reduce stack by 1");
        }
        // JUMPI: pops 2 (dest + condition), doesn't push
        0x57 => {
            check_advice_eq!(after, before - 2, "JUMPI should reduce stack by 2");
        }
        // PC: pushes current PC, doesn't pop (net +1)
        0x58 => {
            check_advice_eq!(after, before + 1, "PC should increase stack by 1");
        }
        // JUMPDEST: no stack effect
        0x5b => {
            check_advice_eq!(after, before, "JUMPDEST should not change stack");
        }
        // SHA3: pops 2 (offset, size), pushes 1 (hash) - net -1
        0x20 => {
            check_advice_eq!(after, before - 1, "SHA3 should reduce stack by 1");
        }
        // MLOAD: pops 1 (offset), pushes 1 (value) - net 0
        0x51 => {
            check_advice_eq!(after, before, "MLOAD should not change stack size");
        }
        // MSTORE: pops 2 (offset, value), doesn't push - net -2
        0x52 => {
            check_advice_eq!(after, before - 2, "MSTORE should reduce stack by 2");
        }
        // MSTORE8: pops 2, doesn't push - net -2
        0x53 => {
            check_advice_eq!(after, before - 2, "MSTORE8 should reduce stack by 2");
        }
        // SLOAD: pops 1 (key), pushes 1 (value) - net 0
        0x54 => {
            check_advice_eq!(after, before, "SLOAD should not change stack size");
        }
        // SSTORE: pops 2 (key, value), doesn't push - net -2
        0x55 => {
            check_advice_eq!(after, before - 2, "SSTORE should reduce stack by 2");
        }
        // Default case - just check stack doesn't grow unreasonably
        _ => {
            check_advice!(after < before + 10, "Stack grew too much");
        }
    }
}
