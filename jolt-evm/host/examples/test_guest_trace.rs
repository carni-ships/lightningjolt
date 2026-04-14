//! Test harness: Generate a trace and run it through the Jolt guest
//!
//! This demonstrates the full Type B1 zkEVM pipeline:
//! 1. Host executes bytecode and produces a trace
//! 2. Trace is serialized to flat binary format
//! 3. Guest verifies the trace using check_advice! constraints

use jolt_evm::{Address, U256, EvmTrace};
use jolt_evm_host::Executor;
use std::io::Write;

/// Trace step in flat binary format (41 bytes)
/// [opcode: u8, pc: u64, gas_before: u64, gas_used: u64, stack_len_before: usize, stack_len_after: usize]
const STEP_SIZE: usize = 41;

fn serialize_trace(trace: &EvmTrace) -> Vec<u8> {
    let mut data = Vec::with_capacity(trace.steps.len() * STEP_SIZE);

    // We need pre-execution stack lengths for each step
    // Track running stack length
    let mut stack_len = 0usize;

    for step in &trace.steps {
        let opcode = step.opcode;
        let pc = step.pc;
        let gas_before = step.gas;
        let gas_used = step.gas_used;

        // Calculate stack_len_before and stack_len_after based on opcode
        let stack_len_before = stack_len;
        let stack_len_after = calculate_stack_len_after(opcode, stack_len_before);

        // Write opcode (1 byte)
        data.push(opcode);

        // Write pc (8 bytes, little-endian)
        data.extend_from_slice(&pc.to_le_bytes());

        // Write gas_before (8 bytes)
        data.extend_from_slice(&gas_before.to_le_bytes());

        // Write gas_used (8 bytes)
        data.extend_from_slice(&gas_used.to_le_bytes());

        // Write stack_len_before (8 bytes)
        data.extend_from_slice(&(stack_len_before as u64).to_le_bytes());

        // Write stack_len_after (8 bytes)
        data.extend_from_slice(&(stack_len_after as u64).to_le_bytes());

        stack_len = stack_len_after;
    }

    data
}

fn calculate_stack_len_after(opcode: u8, before: usize) -> usize {
    match opcode {
        // STOP, REVERT, INVALID: doesn't matter (execution ends)
        0x00 | 0xf3 | 0xfd | 0xfe => before,
        // PUSH1-PUSH32: increases stack by 1
        0x60..=0x7f => before + 1,
        // DUP1-DUP16: duplicates top, increases by 1
        0x80..=0x8f => before + 1,
        // SWAP1-SWAP16: swaps don't change stack size
        0x90..=0x9f => before,
        // ADD, MUL, SUB, DIV, SDIV, MOD, SMOD, ADDMOD, MULMOD, EXP, SIGNEXTEND: net -1
        0x01..=0x0b => before.saturating_sub(1),
        // LT, GT, SLT, SGT, EQ: net -1
        0x10..=0x14 => before.saturating_sub(1),
        // ISZERO: net 0
        0x15 => before,
        // BYTE, SHL, SHR, SAR: net -1
        0x1a..=0x1d => before.saturating_sub(1),
        // SHA3: net -1
        0x20 => before.saturating_sub(1),
        // JUMP: net -1
        0x56 => before.saturating_sub(1),
        // JUMPI: net -2
        0x57 => before.saturating_sub(2),
        // PC: net +1
        0x58 => before + 1,
        // JUMPDEST: no change
        0x5b => before,
        // MLOAD: net 0
        0x51 => before,
        // MSTORE: net -2
        0x52 => before.saturating_sub(2),
        // MSTORE8: net -2
        0x53 => before.saturating_sub(2),
        // SLOAD: net 0
        0x54 => before,
        // SSTORE: net -2
        0x55 => before.saturating_sub(2),
        // Default: assume no change
        _ => before,
    }
}

fn serialize_trace_to_file(trace: &EvmTrace, path: &str) -> std::io::Result<()> {
    let data = serialize_trace(trace);
    let mut file = std::fs::File::create(path)?;
    file.write_all(&data)?;
    Ok(())
}

fn main() {
    println!("=== Type B1 zkEVM Test Harness ===\n");

    // Example 1: Simple ADD program
    // Bytecode: PUSH1 5 PUSH1 3 ADD STOP
    // Stack: [5] -> [5, 3] -> [8]
    let bytecode = vec![0x60, 0x05, 0x60, 0x03, 0x01, 0x00];
    println!("Example 1: PUSH1 5 PUSH1 3 ADD STOP");
    println!("  Bytecode: {:02x?}", bytecode);

    let caller = Address([0u8; 20]);
    let callee = Address([1u8; 20]);
    let value = U256::zero();

    // Execute and generate trace
    let trace = Executor::execute(
        &bytecode,
        0,                      // entry_pc
        caller,
        callee,
        value,
        vec![],                 // input_data
        100000,                 // gas_limit
    ).expect("execution failed");

    println!("\n  Trace generated:");
    println!("    Steps: {}", trace.steps.len());
    println!("    Success: {}", trace.success);
    println!("    Final gas: {}", trace.final_gas);

    // Serialize trace to flat binary
    let trace_data = serialize_trace(&trace);
    println!("    Serialized size: {} bytes ({} steps x {} bytes/step)",
             trace_data.len(), trace.steps.len(), STEP_SIZE);

    // Write trace data to file
    let trace_path = "/tmp/evm_trace.bin";
    serialize_trace_to_file(&trace, trace_path).expect("failed to write trace");
    println!("    Trace written to: {}", trace_path);

    // Print trace info
    println!("    Code hash: 0x{}", hex::encode(trace.code_hash.0));
    println!("    Caller: 0x{}", hex::encode(trace.caller.0));
    println!("    Callee: 0x{}", hex::encode(trace.callee.0));

    println!("\n=== To prove this trace with Jolt ===");
    println!("  1. The guest ELF is at: target/riscv64imac-unknown-none-elf/debug/jolt-evm-guest");
    println!("  2. Run: ./target/release/jolt-emu target/riscv64imac-unknown-none-elf/debug/jolt-evm-guest");
    println!("  3. Currently the guest expects input via postcard serialization");
    println!("  4. A full integration would use Jolt's prover API to provide trace data");

    // Print detailed step info
    println!("\n=== Step Details ===");
    let mut stack_len = 0usize;
    for (i, step) in trace.steps.iter().enumerate() {
        let stack_len_before = stack_len;
        let stack_len_after = calculate_stack_len_after(step.opcode, stack_len_before);
        stack_len = stack_len_after;

        println!("  Step {}: opcode=0x{:02x} pc={} gas_before={} gas_used={} stack: {} -> {}",
                 i, step.opcode, step.pc, step.gas, step.gas_used, stack_len_before, stack_len_after);
    }
}
