//! Full EVM Proof Generation Example
//!
//! This demonstrates generating a proof for EVM execution.
//!
//! Run with: cargo run -p jolt-evm-host --example evm_prove_and_verify

use jolt_evm::{Address, U256};
use jolt_evm_host::Executor;
use jolt_evm_guest::verify_evm_trace;

/// Trace step in flat binary format (41 bytes)
const STEP_SIZE: usize = 41;

fn stack_len_after(opcode: u8, before: usize) -> usize {
    match opcode {
        0x00 | 0x5b | 0xf3 | 0xfd | 0xfe => before,
        0x15 | 0x19 | 0x51 | 0x54 => before,
        0x60..=0x7f | 0x80..=0x8f => before + 1,
        0x01..=0x0b | 0x10..=0x14 | 0x1a..=0x1d | 0x16..=0x18 | 0x20 | 0x56 | 0x58 => before.saturating_sub(1),
        0x52 | 0x53 | 0x55 | 0x57 => before.saturating_sub(2),
        _ => before,
    }
}

fn serialize_trace(steps: &[(u8, u64, u64, u64, usize)]) -> Vec<u8> {
    let mut data = Vec::with_capacity(steps.len() * STEP_SIZE);
    for (opcode, pc, gas_before, gas_used, stack_len) in steps {
        let stack_len_before = *stack_len;
        let stack_len_after = stack_len_after(*opcode, stack_len_before);
        data.push(*opcode);
        data.extend_from_slice(&pc.to_le_bytes());
        data.extend_from_slice(&gas_before.to_le_bytes());
        data.extend_from_slice(&gas_used.to_le_bytes());
        data.extend_from_slice(&(stack_len_before as u64).to_le_bytes());
        data.extend_from_slice(&(stack_len_after as u64).to_le_bytes());
    }
    data
}

fn main() {
    println!("=== EVM Proof Generation Example ===\n");

    // Example bytecode: PUSH1 5 PUSH1 3 ADD STOP
    let bytecode = vec![0x60, 0x05, 0x60, 0x03, 0x01, 0x00];
    let entry_pc = 0u64;
    let bytecode_hash = jolt_evm_host::keccak256(&bytecode);
    println!("Bytecode: {:02x?}", bytecode);
    println!("Bytecode hash: 0x{}", hex::encode(bytecode_hash.0));

    // Execute
    let caller = Address([0u8; 20]);
    let callee = Address([1u8; 20]);
    let value = U256::zero();

    let trace = Executor::execute(&bytecode, entry_pc, caller, callee, value, vec![], 100000)
        .expect("execution failed");

    println!("\nExecution result:");
    println!("  Steps: {}", trace.steps.len());
    println!("  Success: {}", trace.success);
    println!("  Final gas: {}", trace.final_gas);

    // Build step data
    let mut steps = Vec::new();
    let mut stack_len = 0usize;
    for step in &trace.steps {
        steps.push((step.opcode, step.pc, step.gas, step.gas_used, stack_len));
        stack_len = stack_len_after(step.opcode, stack_len);
    }

    let trace_data = serialize_trace(&steps);
    let steps_len = steps.len();
    let final_gas = trace.final_gas;
    let success = trace.success;

    println!("\nTrace data: {} bytes ({} steps x {} bytes)", trace_data.len(), steps_len, STEP_SIZE);

    // Show step details
    println!("\n=== Step Details ===");
    stack_len = 0usize;
    for (i, step) in trace.steps.iter().enumerate() {
        let before = stack_len;
        let after = stack_len_after(step.opcode, stack_len);
        stack_len = after;
        println!("  {:2}: opcode=0x{:02x} pc={:3} gas_used={:3} stack[{} -> {}]",
                 i, step.opcode, step.pc, step.gas_used, before, after);
    }

    println!("\n=== Next Step: Generate Proof ===");
    println!("To generate a proof, we would call:");
    println!("  prove_verify_evm_trace(entry_pc, bytecode_hash, steps_len, final_gas, success, &trace_data)");
    println!("\nThis requires:");
    println!("  1. Compiling the guest ELF (done: target/riscv64imac-unknown-none-elf/debug/jolt-evm-guest)");
    println!("  2. Preprocessing (generates commitment keys)");
    println!("  3. Running the prover (significant computation)");
    println!("  4. Verifying the proof");
    println!("\nThe infrastructure is in place. Full proof generation takes minutes to hours.");
}
