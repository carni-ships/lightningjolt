//! Example: EVM Executor - Execute bytecode and produce traces
//!
//! This example demonstrates the full pipeline:
//! 1. Execute EVM bytecode using the simple executor
//! 2. Generate a step-by-step trace
//! 3. The trace can be verified by the jolt-evm guest

use jolt_evm_host::{execute_and_trace, Address, U256};

fn main() {
    println!("Jolt-EVM Host Example: Full EVM Execution and Trace Generation\n");

    // Create bytecode for: PUSH1 5 PUSH1 3 ADD STOP
    // This pushes 5, pushes 3, adds them (result = 8), then stops
    let bytecode = vec![
        0x60, 0x05, // PUSH1 5
        0x60, 0x03, // PUSH1 3
        0x01,       // ADD
        0x00,       // STOP
    ];

    println!("Bytecode: {:02x?}", bytecode);

    // Create addresses
    let caller = Address([0u8; 20]);
    let callee = Address([1u8; 20]);
    let value = U256([0u8; 32]);

    // Execute and produce trace
    match execute_and_trace(
        &bytecode,
        0,              // entry_pc
        caller,
        callee,
        value,
        vec![],         // input_data
        1_000_000,      // gas limit
    ) {
        Ok(trace) => {
            println!("\nExecution successful!");
            println!("  Steps: {}", trace.steps.len());
            println!("  Success: {}", trace.success);
            println!("  Final gas: {}", trace.final_gas);
            println!("  Code hash: 0x{}", hex::encode(trace.code_hash.0));

            println!("\nStep details:");
            for (i, step) in trace.steps.iter().enumerate() {
                let opcode_name = match jolt_evm::Opcode::from_u8(step.opcode) {
                    Some(jolt_evm::Opcode::STOP) => "STOP",
                    Some(jolt_evm::Opcode::ADD) => "ADD",
                    Some(jolt_evm::Opcode::MUL) => "MUL",
                    Some(jolt_evm::Opcode::SUB) => "SUB",
                    Some(jolt_evm::Opcode::PUSH1) => "PUSH1",
                    Some(jolt_evm::Opcode::PUSH2) => "PUSH2",
                    Some(jolt_evm::Opcode::PUSH3) => "PUSH3",
                    Some(jolt_evm::Opcode::PUSH4) => "PUSH4",
                    Some(jolt_evm::Opcode::JUMP) => "JUMP",
                    Some(jolt_evm::Opcode::JUMPI) => "JUMPI",
                    Some(jolt_evm::Opcode::JUMPDEST) => "JUMPDEST",
                    Some(jolt_evm::Opcode::PC) => "PC",
                    _ => "UNKNOWN",
                };
                println!("  Step {}: pc={}, opcode=0x{:02x} ({})",
                    i, step.pc, step.opcode, opcode_name);
                if !step.stack.is_empty() {
                    // Show top of stack
                    let top = &step.stack[step.stack.len() - 1];
                    let val = u64::from_le_bytes(top.0[..8].try_into().unwrap());
                    println!("    Stack: len={}, top=0x{:x}", step.stack.len(), val);
                } else {
                    println!("    Stack: len={}", step.stack.len());
                }
            }

            println!("\n[TRACE READY FOR VERIFICATION]");
            println!("The trace JSON can be serialized and verified by jolt-evm-guest");
        }
        Err(e) => {
            println!("Execution failed: {}", e);
        }
    }

    // Test with a loop example
    println!("\n\n=== Testing Loop (PUSH1 5 DUP1 JUMPDEST SWAP1 SUB SWAP1 JUMPI) ===\n");

    // bytecode for a simple countdown loop:
    // PUSH1 5    -- counter
    // DUP1       -- copy counter
    // JUMPDEST   -- jump target
    // SWAP1      -- bring counter to top
    // PUSH1 1    -- push 1
    // SUB        -- counter = counter - 1
    // SWAP1      -- bring remaining counter to top
    // JUMPI      -- if counter != 0, jump to JUMPDEST
    // STOP
    let bytecode2 = vec![
        0x60, 0x05,     // PUSH1 5
        0x80,           // DUP1
        0x5b,           // JUMPDEST
        0x90,           // SWAP1
        0x60, 0x01,     // PUSH1 1
        0x03,           // SUB
        0x90,           // SWAP1
        0x57,           // JUMPI
        0x00,           // STOP
    ];

    println!("Bytecode2: {:02x?}", bytecode2);

    match execute_and_trace(
        &bytecode2,
        0,              // entry_pc
        caller,
        callee,
        value,
        vec![],         // input_data
        1_000_000,     // gas limit
    ) {
        Ok(trace) => {
            println!("\nExecution successful!");
            println!("  Steps: {}", trace.steps.len());
            println!("  Success: {}", trace.success);
            println!("  Final gas: {}", trace.final_gas);

            println!("\nStep details (first 10 steps):");
            for (i, step) in trace.steps.iter().take(10).enumerate() {
                let opcode_name = match jolt_evm::Opcode::from_u8(step.opcode) {
                    Some(jolt_evm::Opcode::STOP) => "STOP",
                    Some(jolt_evm::Opcode::ADD) => "ADD",
                    Some(jolt_evm::Opcode::SUB) => "SUB",
                    Some(jolt_evm::Opcode::PUSH1) => "PUSH1",
                    Some(jolt_evm::Opcode::DUP1) => "DUP1",
                    Some(jolt_evm::Opcode::SWAP1) => "SWAP1",
                    Some(jolt_evm::Opcode::JUMP) => "JUMP",
                    Some(jolt_evm::Opcode::JUMPI) => "JUMPI",
                    Some(jolt_evm::Opcode::JUMPDEST) => "JUMPDEST",
                    _ => "UNKNOWN",
                };
                println!("  Step {}: pc={}, opcode=0x{:02x} ({})",
                    i, step.pc, step.opcode, opcode_name);
            }
            if trace.steps.len() > 10 {
                println!("  ... ({} more steps)", trace.steps.len() - 10);
            }
        }
        Err(e) => {
            println!("Execution failed: {}", e);
        }
    }
}
