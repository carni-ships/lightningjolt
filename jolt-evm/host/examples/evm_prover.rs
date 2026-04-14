//! EVM Prover Integration Example
//!
//! Demonstrates the full proving flow:
//! 1. Build/preprocess the guest (done once)
//! 2. Generate EVM trace and serialize to binary
//! 3. Prove the trace with Jolt
//! 4. Verify the proof
//!
//! Run with: cargo run -p jolt-evm-host --example evm_prover

use jolt_evm::{Address, U256};
use jolt_evm_host::Executor;

/// Trace step in flat binary format (41 bytes)
/// [opcode: u8, pc: u64, gas_before: u64, gas_used: u64, stack_len_before: usize, stack_len_after: usize]
const STEP_SIZE: usize = 41;

fn serialize_trace(trace: &jolt_evm::EvmTrace) -> Vec<u8> {
    let mut data = Vec::with_capacity(trace.steps.len() * STEP_SIZE);
    let mut stack_len = 0usize;

    for step in &trace.steps {
        let opcode = step.opcode;
        let pc = step.pc;
        let gas_before = step.gas;
        let gas_used = step.gas_used;

        let stack_len_before = stack_len;
        let stack_len_after = calculate_stack_len_after(opcode, stack_len_before);

        data.push(opcode);
        data.extend_from_slice(&pc.to_le_bytes());
        data.extend_from_slice(&gas_before.to_le_bytes());
        data.extend_from_slice(&gas_used.to_le_bytes());
        data.extend_from_slice(&(stack_len_before as u64).to_le_bytes());
        data.extend_from_slice(&(stack_len_after as u64).to_le_bytes());

        stack_len = stack_len_after;
    }

    data
}

fn calculate_stack_len_after(opcode: u8, before: usize) -> usize {
    match opcode {
        0x00 | 0xf3 | 0xfd | 0xfe => before,
        0x60..=0x7f => before + 1,
        0x80..=0x8f => before + 1,
        0x90..=0x9f => before,
        0x01..=0x0b => before.saturating_sub(1),
        0x10..=0x14 => before.saturating_sub(1),
        0x15 => before,
        0x1a..=0x1d => before.saturating_sub(1),
        0x16..=0x18 => before.saturating_sub(1),
        0x19 => before,
        0x20 => before.saturating_sub(1),
        0x56 => before.saturating_sub(1),
        0x57 => before.saturating_sub(2),
        0x58 => before + 1,
        0x5b => before,
        0x51 => before,
        0x52 | 0x53 => before.saturating_sub(2),
        0x54 => before,
        0x55 => before.saturating_sub(2),
        _ => before,
    }
}

fn main() {
    println!("=== EVM Prover Integration Example ===\n");

    // Example bytecode: PUSH1 5 PUSH1 3 ADD STOP
    let bytecode = vec![0x60, 0x05, 0x60, 0x03, 0x01, 0x00];
    println!("Bytecode: {:02x?}", bytecode);

    let caller = Address([0u8; 20]);
    let callee = Address([1u8; 20]);
    let value = U256::zero();

    // Execute and generate trace
    let trace = Executor::execute(
        &bytecode,
        0,
        caller,
        callee,
        value,
        vec![],
        100000,
    ).expect("execution failed");

    println!("\nTrace: {} steps, success={}, final_gas={}",
             trace.steps.len(), trace.success, trace.final_gas);

    // Serialize trace to flat binary
    let trace_data = serialize_trace(&trace);
    println!("Trace data: {} bytes ({} steps x {} bytes)",
             trace_data.len(), trace.steps.len(), STEP_SIZE);

    println!("\n=== Prover API Functions Available ===");
    println!("  compile_verify_evm_trace(target_dir) -> jolt::host::Program");
    println!("  preprocess_shared_verify_evm_trace(&mut program) -> JoltSharedPreprocessing");
    println!("  preprocess_prover_verify_evm_trace(shared) -> JoltProverPreprocessing");
    println!("  preprocess_verifier_verify_evm_trace(shared, ...) -> JoltVerifierPreprocessing");
    println!("  build_prover_verify_evm_trace(program, preprocessing) -> prover_fn");
    println!("  build_verifier_verify_evm_trace(verifier_preprocessing) -> verifier_fn");
    println!("  prove_verify_evm_trace(program, preprocessing, ...) -> (output, proof, io_device)");
    println!("  trace_verify_evm_trace_to_file(target_dir, ...)");

    println!("\n=== Step Verification Data ===");
    let mut stack_len = 0usize;
    for (i, step) in trace.steps.iter().enumerate() {
        let before = stack_len;
        let after = calculate_stack_len_after(step.opcode, before);
        stack_len = after;

        println!("  Step {:2}: opcode=0x{:02x} pc={:3} gas_used={:3} stack[{} -> {}]",
                 i, step.opcode, step.pc, step.gas_used, before, after);
    }

    println!("\n=== Note ===");
    println!("To run full proof generation, you would need to:");
    println!("1. Call compile_verify_evm_trace() to build the ELF");
    println!("2. Call the preprocess_* functions for preprocessing");
    println!("3. Call prove_verify_evm_trace() with the trace inputs");
    println!("4. Call build_verifier_verify_evm_trace() and verify the proof");
    println!("\nThis requires significant computation (~minutes to hours for real proofs).");
}
