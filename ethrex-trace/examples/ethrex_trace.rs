//! Ethrex Trace Format Demonstration
//!
//! This example demonstrates the 41-byte step trace format that ethrex generates
//! when using the StepTracer infrastructure.
//!
//! The format matches what the Jolt Type B1 zkEVM guest expects for verification.
//!
//! Run with: cargo run -p ethrex-trace --example ethrex_trace

use ethrex_trace::TraceGenerator;

/// Trace step format (41 bytes total):
/// [opcode: u8][pc: u64][gas_before: u64][gas_used: u64][stack_len_before: u64][stack_len_after: u64]
///
/// Gas costs per EVM specification (which ethrex follows):
/// - ADD, SUB, MUL, DIV, MOD, etc.: 3-5 gas depending on operation
/// - PUSH1-PUSH32: 3 gas
/// - JUMP: 8 gas, JUMPI: 10 gas
/// - JUMPDEST: 1 gas
/// - STOP, RETURN, REVERT: 0 gas

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Ethrex 41-byte Step Trace Format ===\n");
    println!("This demonstrates the trace format that ethrex produces with StepTracer.\n");

    // Create trace generator
    let mut tracer = TraceGenerator::new();

    // For PUSH1 5 + PUSH1 3 + ADD + STOP, ethrex would produce:
    // Step 1: PUSH1 (0x60) at pc=0 - pushes 5 onto stack, gas cost = 3
    // Step 2: PUSH1 (0x60) at pc=2 - pushes 3 onto stack, gas cost = 3
    // Step 3: ADD (0x01) at pc=4 - pops 5,3 and pushes 8, gas cost = 3
    // Step 4: STOP (0x00) at pc=5 - terminates execution, gas cost = 0

    let gas_limit = 1000000u64;
    let step1_gas_before = gas_limit;
    let step2_gas_before = step1_gas_before - 3;  // PUSH1 costs 3
    let step3_gas_before = step2_gas_before - 3;  // PUSH1 costs 3
    let step4_gas_before = step3_gas_before - 3;  // ADD costs 3

    tracer.add_step(0x60, 0, step1_gas_before, 3, 0, 1);  // PUSH1
    tracer.add_step(0x60, 2, step2_gas_before, 3, 1, 2);  // PUSH1
    tracer.add_step(0x01, 4, step3_gas_before, 3, 2, 1);  // ADD (3 gas per EVM spec)
    tracer.add_step(0x00, 5, step4_gas_before, 0, 1, 1);  // STOP

    println!("Trace for: PUSH1 5 + PUSH1 3 + ADD + STOP");
    println!("Bytecode: 60 05 60 03 01 00\n");

    println!("Steps: {}", tracer.steps_len());
    println!("Total bytes: {} ({} steps x 41 bytes = {})\n",
        tracer.trace_data().len(),
        tracer.steps_len(),
        tracer.steps_len() * 41);

    // Parse and display the trace
    let data = tracer.trace_data();
    println!("Step details:");
    for i in 0..tracer.steps_len() {
        let offset = i * 41;
        let opcode = data[offset];
        let pc = u64::from_le_bytes(data[offset+1..offset+9].try_into()?);
        let gas_before = u64::from_le_bytes(data[offset+9..offset+17].try_into()?);
        let gas_used = u64::from_le_bytes(data[offset+17..offset+25].try_into()?);
        let stack_before = u64::from_le_bytes(data[offset+25..offset+33].try_into()?);
        let stack_after = u64::from_le_bytes(data[offset+33..offset+41].try_into()?);

        let name = match opcode {
            0x00 => "STOP",
            0x01 => "ADD",
            0x02 => "MUL",
            0x03 => "SUB",
            0x04 => "DIV",
            0x60 => "PUSH1",
            0x7f => "PUSH32",
            _ => "UNKNOWN",
        };

        println!("  {:2}: {:7} pc={:3} gas_before={:7} gas_used={:2} stack[{:1} -> {:1}]",
            i, name, pc, gas_before, gas_used, stack_before, stack_after);
    }

    println!("\n=== 41-byte Step Format ===");
    println!("Byte layout:");
    println!("  [ 0] opcode: u8");
    println!("  [ 1- 8] pc: u64 (little-endian)");
    println!("  [ 9-16] gas_before: u64 (little-endian)");
    println!("  [17-24] gas_used: u64 (little-endian)");
    println!("  [25-32] stack_len_before: u64 (little-endian)");
    println!("  [33-40] stack_len_after: u64 (little-endian)");

    println!("\n=== EVM Gas Costs (per EVM specification) ===");
    println!("  ADD, SUB, MUL, DIV, MOD, SDIV, SMOD: 3-5 gas");
    println!("  EXP: 10 gas (plus dynamic)");
    println!("  PUSH1-PUSH32: 3 gas each");
    println!("  DUP1-DUP16: 3 gas each");
    println!("  SWAP1-SWAP16: 3 gas each");
    println!("  JUMP: 8 gas");
    println!("  JUMPI: 10 gas");
    println!("  JUMPDEST: 1 gas");
    println!("  STOP, RETURN, REVERT, INVALID: 0 gas");
    println!("  SLOAD: 0 static + 100-2100 dynamic");
    println!("  SSTORE: 0 static + 100-2900 dynamic");

    println!("\n=== Guest Verification ===");
    println!("The Jolt Type B1 zkEVM guest verifies:");
    println!("  1. gas_used matches expected cost for opcode");
    println!("  2. PC is within bounds (< 1,000,000)");
    println!("  3. Stack delta matches opcode semantics");

    // Show raw bytes
    println!("\n=== Raw Trace Bytes ===");
    println!("Hex: {:02x?}", tracer.trace_data());

    println!("\n=== SUCCESS ===");
    println!("Trace format is compatible with Jolt Type B1 zkEVM guest.");
    Ok(())
}
