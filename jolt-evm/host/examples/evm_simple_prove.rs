//! Simple EVM Proof Example
//!
//! Demonstrates the full Type B1 zkEVM flow:
//! 1. Execute bytecode with our EVM
//! 2. Generate step-by-step trace
//! 3. Serialize trace to our 41-byte/step format
//! 4. (Future) Generate proof with Jolt
//!
//! Run with: cargo run -p jolt-evm-host --example evm_simple_prove

use jolt_evm::{Address, H256, U256};
use jolt_evm_host::{Executor, keccak256};

/// Trace step in flat binary format (41 bytes)
/// [opcode: u8, pc: u64, gas_before: u64, gas_used: u64, stack_len_before: usize, stack_len_after: usize]
const STEP_SIZE: usize = 41;

/// Calculate stack length after an opcode executes
fn stack_len_after(opcode: u8, before: usize) -> usize {
    match opcode {
        // Opcodes that don't push/pop (net 0)
        0x00 | 0x5b | 0xf3 | 0xfd | 0xfe => before, // STOP, JUMPDEST, RETURN, REVERT, INVALID
        0x15 | 0x19 | 0x51 | 0x54 => before,      // ISZERO, NOT, MLOAD, SLOAD
        // Opcodes that push (increase by 1)
        0x60..=0x7f | 0x80..=0x8f => before + 1,   // PUSH*, DUP*
        // Opcodes that pop only (decrease by 1)
        0x01..=0x0b | 0x10..=0x14 | 0x1a..=0x1d | 0x16..=0x18 | 0x20 | 0x56 | 0x58 => before.saturating_sub(1),
        // Opcodes that pop 2 (decrease by 2)
        0x52 | 0x53 | 0x55 | 0x57 => before.saturating_sub(2),
        _ => before,
    }
}

/// Serialize an EVM trace to our flat binary format
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

/// Execute bytecode and return trace data
fn execute_and_trace(bytecode: &[u8]) -> (Vec<u8>, usize, u64, bool) {
    let caller = Address([0u8; 20]);
    let callee = Address([1u8; 20]);
    let value = U256::zero();

    // Use our simple executor
    let trace = Executor::execute(
        bytecode,
        0,
        caller,
        callee,
        value,
        vec![],
        100000,
    ).expect("execution failed");

    // Build step data from trace
    let mut steps = Vec::new();
    let mut stack_len = 0usize;

    for step in &trace.steps {
        let opcode = step.opcode;
        let pc = step.pc;
        let gas_before = step.gas;
        let gas_used = step.gas_used;

        steps.push((opcode, pc, gas_before, gas_used, stack_len));
        stack_len = stack_len_after(opcode, stack_len);
    }

    let trace_data = serialize_trace(&steps);
    let steps_len = steps.len();
    let final_gas = trace.final_gas;
    let success = trace.success;

    (trace_data, steps_len, final_gas, success)
}

fn main() {
    println!("=== Simple EVM Proof Example ===\n");

    // Example 1: Simple addition
    // Bytecode: PUSH1 5 PUSH1 3 ADD STOP
    // Stack: [5] -> [5, 3] -> [8]
    let bytecode1 = vec![0x60, 0x05, 0x60, 0x03, 0x01, 0x00];
    println!("Example 1: PUSH1 5 + PUSH1 3 + ADD");
    println!("Bytecode: {:02x?}", bytecode1);

    let (trace_data1, steps_len1, final_gas1, success1) = execute_and_trace(&bytecode1);
    println!("Trace: {} steps, success={}, final_gas={}", steps_len1, success1, final_gas1);
    println!("Trace data: {} bytes\n", trace_data1.len());

    // Example 2: Multiplication
    // Bytecode: PUSH1 6 PUSH1 5 MUL STOP
    let bytecode2 = vec![0x60, 0x06, 0x60, 0x05, 0x02, 0x00];
    println!("Example 2: PUSH1 6 * PUSH1 5 + MUL");
    println!("Bytecode: {:02x?}", bytecode2);

    let (trace_data2, steps_len2, final_gas2, success2) = execute_and_trace(&bytecode2);
    println!("Trace: {} steps, success={}, final_gas={}", steps_len2, success2, final_gas2);
    println!("Trace data: {} bytes\n", trace_data2.len());

    // Example 3: Simple storage access
    // Bytecode: PUSH1 0 SLOAD STOP  (reads slot 0, returns 0)
    let bytecode3 = vec![0x60, 0x00, 0x54, 0x00];
    println!("Example 3: SLOAD (read storage slot 0)");
    println!("Bytecode: {:02x?}", bytecode3);

    let (trace_data3, steps_len3, final_gas3, success3) = execute_and_trace(&bytecode3);
    println!("Trace: {} steps, success={}, final_gas={}", steps_len3, success3, final_gas3);
    println!("Trace data: {} bytes\n", trace_data3.len());

    // Show detailed step info for example 1
    println!("=== Example 1 Step Details ===");
    let caller = Address([0u8; 20]);
    let callee = Address([1u8; 20]);
    let value = U256::zero();
    let trace = Executor::execute(&bytecode1, 0, caller, callee, value, vec![], 100000).expect("failed");
    let mut stack_len = 0usize;
    for (i, step) in trace.steps.iter().enumerate() {
        let stack_before = stack_len;
        let stack_after = stack_len_after(step.opcode, stack_len);
        stack_len = stack_after;
        println!("  Step {:2}: opcode=0x{:02x} pc={:3} gas_before={:6} gas_used={:3} stack[{} -> {}]",
                 i, step.opcode, step.pc, step.gas, step.gas_used, stack_before, stack_after);
    }

    println!("\n=== Summary ===");
    println!("We have working:");
    println!("  1. EVM executor that generates traces");
    println!("  2. Trace serialization to our format");
    println!("  3. Guest that verifies traces (check_advice!)");
    println!("  4. ELF that runs on RISC-V");
    println!("\nNext steps:");
    println!("  1. Integrate prover API to generate proofs");
    println!("  2. Extend EVM with missing opcodes");
    println!("  3. Add state management for full blocks");
}
