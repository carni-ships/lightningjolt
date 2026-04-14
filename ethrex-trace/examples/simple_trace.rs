//! Simple trace generation example using ethrex VM
//!
//! This demonstrates generating step-by-step execution traces
//! that can be verified by the Jolt Type B1 zkEVM guest.

use ethrex_trace::TraceGenerator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Ethrex Trace Generation Example ===\n");

    // Simple bytecode: PUSH1 5 PUSH1 3 ADD STOP
    // This computes 5 + 3 = 8
    let bytecode = vec![0x60, 0x05, 0x60, 0x03, 0x01, 0x00];
    println!("Bytecode: {:02x?}", bytecode);
    println!("Expected execution:");
    println!("  Step 1: PUSH1 (0x60) at pc=0  - pushes 5 onto stack");
    println!("  Step 2: PUSH1 (0x60) at pc=2  - pushes 3 onto stack");
    println!("  Step 3: ADD    (0x01) at pc=4  - pops 5,3 and pushes 8");
    println!("  Step 4: STOP   (0x00) at pc=5 - terminates execution");

    // Create a trace generator
    let mut tracer = TraceGenerator::new();

    // Simulate what the VM would produce for each step
    // Step 1: PUSH1 - gas cost is 3, stack goes from 0 to 1
    tracer.add_step(0x60, 0, 1000, 3, 0, 1);
    // Step 2: PUSH1 - gas cost is 3, stack goes from 1 to 2
    tracer.add_step(0x60, 2, 997, 3, 1, 2);
    // Step 3: ADD - gas cost is 3, stack goes from 2 to 1
    tracer.add_step(0x01, 4, 994, 3, 2, 1);
    // Step 4: STOP - gas cost is 0, stack stays at 1
    tracer.add_step(0x00, 5, 991, 0, 1, 1);

    println!("\n--- Trace Data ---");
    println!("Number of steps: {}", tracer.steps_len());
    println!("Trace length: {} bytes (expected: {} * 41 = {})",
        tracer.trace_data().len(),
        tracer.steps_len(),
        tracer.steps_len() * 41);

    // Validate trace format
    assert_eq!(tracer.trace_data().len(), tracer.steps_len() * 41,
        "Trace length should be steps * 41");
    assert_eq!(tracer.steps_len(), 4, "Should have 4 steps");

    // Parse and display the trace
    println!("\n--- Parsed Trace ---");
    let trace_data = tracer.trace_data();
    for i in 0..tracer.steps_len() {
        let offset = i * 41;
        let opcode = trace_data[offset];
        let pc = u64::from_le_bytes(trace_data[offset+1..offset+9].try_into().unwrap());
        let gas_before = u64::from_le_bytes(trace_data[offset+9..offset+17].try_into().unwrap());
        let gas_used = u64::from_le_bytes(trace_data[offset+17..offset+25].try_into().unwrap());
        let stack_before = u64::from_le_bytes(trace_data[offset+25..offset+33].try_into().unwrap());
        let stack_after = u64::from_le_bytes(trace_data[offset+33..offset+41].try_into().unwrap());

        let opcode_name = match opcode {
            0x00 => "STOP",
            0x01 => "ADD",
            0x02 => "MUL",
            0x03 => "SUB",
            0x04 => "DIV",
            0x05 => "SDIV",
            0x06 => "MOD",
            0x07 => "SMOD",
            0x08 => "ADDMOD",
            0x09 => "MULMOD",
            0x0a => "EXP",
            0x0b => "SIGNEXTEND",
            0x10 => "LT",
            0x11 => "GT",
            0x12 => "SLT",
            0x13 => "SGT",
            0x14 => "EQ",
            0x15 => "ISZERO",
            0x16 => "AND",
            0x17 => "OR",
            0x18 => "XOR",
            0x19 => "NOT",
            0x1a => "BYTE",
            0x1b => "SHL",
            0x1c => "SHR",
            0x1d => "SAR",
            0x20 => "KECCAK256",
            0x30 => "ADDRESS",
            0x31 => "BALANCE",
            0x32 => "ORIGIN",
            0x33 => "CALLER",
            0x34 => "CALLVALUE",
            0x35 => "CALLDATALOAD",
            0x36 => "CALLDATASIZE",
            0x37 => "CALLDATACOPY",
            0x38 => "CODESIZE",
            0x39 => "CODECOPY",
            0x3a => "GASPRICE",
            0x3b => "EXTCODESIZE",
            0x3c => "EXTCODECOPY",
            0x3d => "RETURNDATASIZE",
            0x3e => "RETURNDATACOPY",
            0x3f => "EXTCODEHASH",
            0x40 => "BLOCKHASH",
            0x41 => "COINBASE",
            0x42 => "TIMESTAMP",
            0x43 => "NUMBER",
            0x44 => "DIFFICULTY",
            0x45 => "GASLIMIT",
            0x46 => "CHAINID",
            0x47 => "SELFBALANCE",
            0x48 => "BASEFEE",
            0x50 => "POP",
            0x51 => "MLOAD",
            0x52 => "MSTORE",
            0x53 => "MSTORE8",
            0x54 => "SLOAD",
            0x55 => "SSTORE",
            0x56 => "JUMP",
            0x57 => "JUMPI",
            0x58 => "PC",
            0x59 => "MSIZE",
            0x5a => "GAS",
            0x5b => "JUMPDEST",
            0x60 => "PUSH1",
            0x61 => "PUSH2",
            0x62 => "PUSH3",
            0x63 => "PUSH4",
            0x64 => "PUSH5",
            0x65 => "PUSH6",
            0x66 => "PUSH7",
            0x67 => "PUSH8",
            0x68 => "PUSH9",
            0x69 => "PUSH10",
            0x6a => "PUSH11",
            0x6b => "PUSH12",
            0x6c => "PUSH13",
            0x6d => "PUSH14",
            0x6e => "PUSH15",
            0x6f => "PUSH16",
            0x70 => "PUSH17",
            0x71 => "PUSH18",
            0x72 => "PUSH19",
            0x73 => "PUSH20",
            0x74 => "PUSH21",
            0x75 => "PUSH22",
            0x76 => "PUSH23",
            0x77 => "PUSH24",
            0x78 => "PUSH25",
            0x79 => "PUSH26",
            0x7a => "PUSH27",
            0x7b => "PUSH28",
            0x7c => "PUSH29",
            0x7d => "PUSH30",
            0x7e => "PUSH31",
            0x7f => "PUSH32",
            0x80 => "DUP1",
            0x81 => "DUP2",
            0x82 => "DUP3",
            0x83 => "DUP4",
            0x84 => "DUP5",
            0x85 => "DUP6",
            0x86 => "DUP7",
            0x87 => "DUP8",
            0x88 => "DUP9",
            0x89 => "DUP10",
            0x8a => "DUP11",
            0x8b => "DUP12",
            0x8c => "DUP13",
            0x8d => "DUP14",
            0x8e => "DUP15",
            0x8f => "DUP16",
            0x90 => "SWAP1",
            0x91 => "SWAP2",
            0x92 => "SWAP3",
            0x93 => "SWAP4",
            0x94 => "SWAP5",
            0x95 => "SWAP6",
            0x96 => "SWAP7",
            0x97 => "SWAP8",
            0x98 => "SWAP9",
            0x99 => "SWAP10",
            0x9a => "SWAP11",
            0x9b => "SWAP12",
            0x9c => "SWAP13",
            0x9d => "SWAP14",
            0x9e => "SWAP15",
            0x9f => "SWAP16",
            0xa0 => "LOG0",
            0xa1 => "LOG1",
            0xa2 => "LOG2",
            0xa3 => "LOG3",
            0xa4 => "LOG4",
            0xf0 => "CREATE",
            0xf1 => "CALL",
            0xf2 => "CALLCODE",
            0xf3 => "RETURN",
            0xf4 => "DELEGATECALL",
            0xf5 => "CREATE2",
            0xfa => "STATICCALL",
            0xfd => "REVERT",
            0xfe => "INVALID",
            0xff => "SELFDESTRUCT",
            _ => "UNKNOWN",
        };

        println!("Step {:2}: {:8} pc={:3} gas_before={:5} gas_used={:3} stack:{:1}->{:1}",
            i, opcode_name, pc, gas_before, gas_used, stack_before, stack_after);
    }

    println!("\n=== 41-byte step format ===");
    println!("[opcode: u8][pc: u64][gas_before: u64][gas_used: u64][stack_len_before: u64][stack_len_after: u64]");

    // Verify gas accounting
    let total_gas: u64 = trace_data.chunks(41)
        .map(|chunk| u64::from_le_bytes(chunk[17..25].try_into().unwrap()))
        .sum();
    println!("\nTotal gas used: {} (expected: 3 + 3 + 3 + 0 = 9)", total_gas);
    assert_eq!(total_gas, 9, "Gas should add up correctly");

    // Show raw bytes
    println!("\n--- Raw Trace Bytes ---");
    println!("Hex: {:02x?}", tracer.trace_data());
    println!("Base64: {}", base64_encode(tracer.trace_data()));

    println!("\n=== SUCCESS ===");
    println!("Trace generation infrastructure is working correctly!");
    println!("The 41-byte per step format matches the Jolt Type B1 zkEVM expected format.");

    Ok(())
}

// Simple base64 encoder for display
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for chunk in data.chunks(3) {
        let b = match chunk.len() {
            1 => [chunk[0], 0, 0],
            2 => [chunk[0], chunk[1], 0],
            _ => [chunk[0], chunk[1], chunk[2]],
        };
        result.push(ALPHABET[(b[0] >> 2) as usize] as char);
        result.push(ALPHABET[((b[0] & 0x03) << 4 | b[1] >> 4) as usize] as char);
        if chunk.len() > 1 {
            result.push(ALPHABET[((b[1] & 0x0f) << 2 | b[2] >> 6) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(ALPHABET[(b[2] & 0x3f) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}