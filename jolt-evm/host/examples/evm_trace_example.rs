//! Example: Simple EVM trace generation
//!
//! This example demonstrates how to generate an EVM execution trace
//! for verification by the jolt-evm guest.

use jolt_evm::{Address, H256, U256, EvmStep, EvmTrace};
use jolt_evm_host::{generate_trace_from_steps, make_add_step, make_push_step, make_stop_step, keccak256};

fn main() {
    println!("Jolt-EVM Host Example: Generating a simple addition trace\n");

    // Create a simple trace for: PUSH1 5 PUSH1 3 ADD STOP
    // This pushes 5, pushes 3, adds them (result = 8), then stops

    let caller = Address([0u8; 20]);
    let callee = Address([1u8; 20]);
    let value = U256::zero();
    let code_hash = keccak256(b"\x60\x05\x60\x03\x01\x00"); // PUSH1 5 PUSH1 3 ADD STOP

    // Gas tracking
    let gas_limit = 100000u64;
    let gas_per_step = 2u64;

    // Step 1: PUSH1 5 (PC=0)
    let step1 = make_push_step(0, gas_limit, gas_per_step, u256_from_u64(5));

    // Step 2: PUSH1 3 (PC=2)
    let step2 = make_push_step(2, gas_limit - gas_per_step, gas_per_step, u256_from_u64(3));

    // Step 3: ADD (PC=4) - pops 5 and 3, pushes 8
    let step3 = make_add_step(
        4,
        gas_limit - 2 * gas_per_step,
        gas_per_step,
        u256_from_u64(5),
        u256_from_u64(3),
        u256_from_u64(8),
    );

    // Step 4: STOP (PC=5)
    let step4 = make_stop_step(5, gas_limit - 3 * gas_per_step, 0);

    let steps = vec![step1, step2, step3, step4];

    let trace = generate_trace_from_steps(
        0,               // entry_pc
        code_hash,        // code_hash
        caller,           // caller
        callee,          // callee
        value,           // value
        vec![],          // input_data
        steps,
        gas_limit - 3 * gas_per_step, // final_gas
        true,            // success
        vec![],          // return_data
    );

    println!("Generated trace:");
    println!("  Code hash: 0x{}", hex::encode(trace.code_hash.0));
    println!("  Caller: 0x{}", hex::encode(trace.caller.0));
    println!("  Callee: 0x{}", hex::encode(trace.callee.0));
    println!("  Steps: {}", trace.steps.len());
    println!("  Success: {}", trace.success);
    println!("  Final gas: {}", trace.final_gas);
    println!();

    // Serialize the trace to JSON for inspection
    let json = serde_json::to_string_pretty(&trace).unwrap();
    println!("Trace JSON:\n{}", json);

    println!("\nNote: To actually prove this trace with Jolt, you would:");
    println!("  1. Build the guest ELF with: jolt build -p jolt-evm-guest");
    println!("  2. Run the host binary to generate and verify the proof");
    println!("  3. The guest verifies each step using check_advice!");
}

/// Helper function to create U256 from u64
fn u256_from_u64(v: u64) -> U256 {
    let mut bytes = [0u8; 32];
    bytes[0] = v as u8;
    bytes[1] = (v >> 8) as u8;
    bytes[2] = (v >> 16) as u8;
    bytes[3] = (v >> 24) as u8;
    bytes[4] = (v >> 32) as u8;
    bytes[5] = (v >> 40) as u8;
    bytes[6] = (v >> 48) as u8;
    bytes[7] = (v >> 56) as u8;
    U256(bytes)
}
