//! Host-side EVM trace generator
//!
//! This module provides utilities for generating EVM execution traces
//! that can be verified by the jolt-evm guest.

// Re-export types from jolt-evm for convenience
pub use jolt_evm::Address;
pub use jolt_evm::H256;
pub use jolt_evm::U256;
pub use jolt_evm::EvmStep;
pub use jolt_evm::EvmTrace;
pub use jolt_evm::Opcode;

pub mod evm;
pub use evm::Executor;

/// Generate a trace from manual step data
///
/// This is useful for testing and for integrating with external EVM implementations.
pub fn generate_trace_from_steps(
    entry_pc: u64,
    code_hash: H256,
    caller: Address,
    callee: Address,
    value: U256,
    input_data: Vec<u8>,
    steps: Vec<EvmStep>,
    final_gas: u64,
    success: bool,
    return_data: Vec<u8>,
) -> EvmTrace {
    EvmTrace {
        entry_pc,
        code_hash,
        caller,
        callee,
        value,
        input_data,
        steps,
        final_gas,
        success,
        return_data,
    }
}

/// Create a simple ADD trace step
pub fn make_add_step(
    pc: u64,
    gas: u64,
    gas_used: u64,
    a: U256,
    b: U256,
    result: U256,
) -> EvmStep {
    EvmStep {
        pc,
        opcode: 0x01, // ADD
        gas,
        gas_used,
        stack: vec![a, b, result],
        memory: None,
        storage_key: None,
        storage_value_before: None,
        storage_value_after: None,
        return_data: None,
    }
}

/// Create a simple PUSH1 trace step
pub fn make_push_step(pc: u64, gas: u64, gas_used: u64, value: U256) -> EvmStep {
    EvmStep {
        pc,
        opcode: 0x60, // PUSH1
        gas,
        gas_used,
        stack: vec![value],
        memory: None,
        storage_key: None,
        storage_value_before: None,
        storage_value_after: None,
        return_data: None,
    }
}

/// Create a STOP step
pub fn make_stop_step(pc: u64, gas: u64, gas_used: u64) -> EvmStep {
    EvmStep {
        pc,
        opcode: 0x00, // STOP
        gas,
        gas_used,
        stack: vec![],
        memory: None,
        storage_key: None,
        storage_value_before: None,
        storage_value_after: None,
        return_data: None,
    }
}

/// Compute keccak256 hash
pub fn keccak256(data: &[u8]) -> H256 {
    use tiny_keccak::Hasher;
    let mut hasher = tiny_keccak::Keccak::v256();
    let mut output = [0u8; 32];
    hasher.update(data);
    hasher.finalize(&mut output);
    H256(output)
}

/// Execute bytecode and produce a trace using the simple EVM executor
pub fn execute_and_trace(
    bytecode: &[u8],
    entry_pc: u64,
    caller: Address,
    callee: Address,
    value: U256,
    input_data: Vec<u8>,
    gas_limit: u64,
) -> Result<EvmTrace, &'static str> {
    Executor::execute(bytecode, entry_pc, caller, callee, value, input_data, gas_limit)
}
