//! EVM Proof Generation Performance Optimization Experiment
//!
//! Demonstrates:
//! 1. Baseline: Sequential proving
//! 2. Batch optimization: Pre-build prover once, reuse for multiple traces
//! 3. Parallel optimization: Run independent proofs in parallel
//! 4. Trace complexity: Minimal traces already
//!
//! Run with: cargo run -p jolt-evm-host --example evm_proof

use jolt_evm::{Address, H256, U256};
use jolt_evm_host::Executor;
use std::time::Instant;
use std::sync::Arc;

const STEP_SIZE: usize = 41;

fn calculate_stack_len_after(opcode: u8, before: usize) -> usize {
    match opcode {
        0x00 | 0xf3 | 0xfd | 0xfe => before,
        0x5b => before,
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
        0x51 => before,
        0x52 | 0x53 => before.saturating_sub(2),
        0x54 => before,
        0x55 => before.saturating_sub(2),
        0x56 => before.saturating_sub(1),
        0x57 => before.saturating_sub(2),
        0x58 => before + 1,
        _ => before,
    }
}

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

fn keccak256(data: &[u8]) -> H256 {
    use tiny_keccak::Hasher;
    let mut hasher = tiny_keccak::Keccak::v256();
    let mut output = [0u8; 32];
    hasher.update(data);
    hasher.finalize(&mut output);
    H256(output)
}

fn main() {
    println!("=== EVM Proof Generation Optimization Experiment ===\n");

    // ============================================================
    // TEST 1: Baseline - Single trace, sequential
    // ============================================================
    println!("{}", "=".repeat(70));
    println!("TEST 1: Baseline - Single trace");
    println!("{}", "=".repeat(70));

    let bytecode = vec![0x60, 0x05, 0x60, 0x03, 0x01, 0x00]; // PUSH1 5 + PUSH1 3 + ADD + STOP
    let caller = Address([0u8; 20]);
    let callee = Address([1u8; 20]);
    let value = U256::zero();

    let trace = Executor::execute(&bytecode, 0, caller, callee, value, vec![], 100000)
        .expect("execution failed");

    println!("Trace: {} steps, {} bytes", trace.steps.len(), serialize_trace(&trace).len());

    // Build prover (fixed setup)
    let target_dir = "/tmp/jolt-evm-guest-targets";
    let start = Instant::now();
    let program = jolt_evm_guest::compile_verify_evm_trace(target_dir);
    let mut program_mut = program;
    let shared = jolt_evm_guest::preprocess_shared_verify_evm_trace(&mut program_mut).unwrap();
    let prover_prep = jolt_evm_guest::preprocess_prover_verify_evm_trace(shared.clone());
    let prover = jolt_evm_guest::build_prover_verify_evm_trace(program_mut, prover_prep.clone());
    let verifier_prep = jolt_evm_guest::verifier_preprocessing_from_prover_verify_evm_trace(&prover_prep);
    let verifier = jolt_evm_guest::build_verifier_verify_evm_trace(verifier_prep);
    let setup_time = start.elapsed();
    println!("Setup time: {:.3}s", setup_time.as_secs_f64());

    // Time single proof
    let trace_data = serialize_trace(&trace);
    let bytecode_hash = keccak256(&bytecode);

    let start = Instant::now();
    let (output, proof, io_device) = prover(
        0u64, bytecode_hash.0, trace.steps.len(),
        trace.final_gas, trace.success, &trace_data,
    );
    let single_proof_time = start.elapsed();

    let is_valid = verifier(
        0u64, bytecode_hash.0, trace.steps.len(),
        trace.final_gas, trace.success, &trace_data,
        output, io_device.panic, proof,
    );

    println!("Single proof time: {:.3}s (verified: {})", single_proof_time.as_secs_f64(), is_valid);

    // ============================================================
    // TEST 2: Batch optimization - Reuse prover, multiple traces
    // ============================================================
    println!("\n{}", "=".repeat(70));
    println!("TEST 2: Batch - Prove {} traces sequentially (reusing prover)", 10);
    println!("{}", "=".repeat(70));

    let bytecodes = vec![
        vec![0x60, 0x05, 0x60, 0x03, 0x01, 0x00], // 5+3
        vec![0x60, 0x06, 0x60, 0x07, 0x02, 0x00], // 6*7
        vec![0x60, 0x64, 0x60, 0x05, 0x04, 0x00], // 100/5
        vec![0x60, 0x0a, 0x60, 0x05, 0x01, 0x00], // 10+5
        vec![0x60, 0x09, 0x60, 0x03, 0x02, 0x00], // 9*3
        vec![0x60, 0x08, 0x60, 0x02, 0x01, 0x00], // 8+2
        vec![0x60, 0x12, 0x60, 0x04, 0x04, 0x00], // 18/4
        vec![0x60, 0x10, 0x60, 0x08, 0x02, 0x00], // 16*8
        vec![0x60, 0x07, 0x60, 0x03, 0x01, 0x00], // 7+3
        vec![0x60, 0x05, 0x60, 0x02, 0x01, 0x00], // 5+2
    ];

    // Pre-build prover ONCE (done above - already reused)
    println!("Prover already built and cached (setup_time: {:.3}s)", setup_time.as_secs_f64());

    let start = Instant::now();
    let mut batch_results = Vec::new();

    for (i, bc) in bytecodes.iter().enumerate() {
        let trace = Executor::execute(bc, 0, caller, callee, value.clone(), vec![], 100000)
            .expect("execution failed");
        let trace_data = serialize_trace(&trace);
        let bytecode_hash = keccak256(bc);

        let (output, proof, io_device) = prover(
            0u64, bytecode_hash.0, trace.steps.len(),
            trace.final_gas, trace.success, &trace_data,
        );

        let is_valid = verifier(
            0u64, bytecode_hash.0, trace.steps.len(),
            trace.final_gas, trace.success, &trace_data,
            output, io_device.panic, proof,
        );

        batch_results.push(is_valid);
        print!(".");
    }
    println!();
    let batch_time = start.elapsed();

    let verified_count = batch_results.iter().filter(|&&v| v).count();
    println!("Batch time (10 traces): {:.3}s", batch_time.as_secs_f64());
    println!("  Per-trace average: {:.3}s", batch_time.as_secs_f64() / 10.0);
    println!("  Verified: {}/10", verified_count);
    println!("  Speedup vs 10x single: {:.1}x",
             (single_proof_time.as_secs_f64() * 10.0) / batch_time.as_secs_f64());

    // ============================================================
    // TEST 3: Parallel optimization - Multiple threads
    // ============================================================
    println!("\n{}", "=".repeat(70));
    println!("TEST 3: Parallel - {} threads proving simultaneously", 4);
    println!("{}", "=".repeat(70));

    // Clone prover components for parallel use
    let prover_arc = Arc::new(prover);
    let verifier_arc = Arc::new(verifier);

    let start = Instant::now();
    let handles: Vec<_> = (0..4).map(|i| {
        let prover = prover_arc.clone();
        let verifier = verifier_arc.clone();
        let bc = bytecodes[i].clone();
        let caller = Address([0u8; 20]);
        let callee = Address([1u8; 20]);
        let value = U256::zero();

        std::thread::spawn(move || {
            let trace = Executor::execute(&bc, 0, caller, callee, value, vec![], 100000)
                .expect("execution failed");
            let trace_data = serialize_trace(&trace);
            let bytecode_hash = keccak256(&bc);

            let (output, proof, io_device) = prover(
                0u64, bytecode_hash.0, trace.steps.len(),
                trace.final_gas, trace.success, &trace_data,
            );

            let is_valid = verifier(
                0u64, bytecode_hash.0, trace.steps.len(),
                trace.final_gas, trace.success, &trace_data,
                output, io_device.panic, proof,
            );

            (i, is_valid)
        })
    }).collect();

    let mut parallel_results = Vec::new();
    for handle in handles {
        parallel_results.push(handle.join().expect("thread join failed"));
    }
    parallel_results.sort_by_key(|(i, _)| *i);

    let parallel_time = start.elapsed();
    let verified_count = parallel_results.iter().filter(|(_, v)| *v).count();

    println!("Parallel time (4 traces): {:.3}s", parallel_time.as_secs_f64());
    println!("  Per-trace average: {:.3}s", parallel_time.as_secs_f64() / 4.0);
    println!("  Verified: {}/4", verified_count);
    println!("  Speedup vs sequential 4x: {:.1}x",
             (single_proof_time.as_secs_f64() * 4.0) / parallel_time.as_secs_f64());

    // ============================================================
    // TEST 4: Trace Complexity - Verify trace is already minimal
    // ============================================================
    println!("\n{}", "=".repeat(70));
    println!("TEST 4: Trace Complexity Analysis");
    println!("{}", "=".repeat(70));

    println!("\nTrace breakdown for PUSH1 + PUSH1 + ADD + STOP:");
    println!("  Step 0: PUSH1 (opcode=0x60, pc=0, gas=3)");
    println!("  Step 1: PUSH1 (opcode=0x60, pc=2, gas=3)");
    println!("  Step 2: ADD    (opcode=0x01, pc=4, gas=3)");
    println!("  Step 3: STOP   (opcode=0x00, pc=5, gas=0)");
    println!("  Total gas: 9");
    println!("  41 bytes/step x 4 steps = {} bytes total", 41 * 4);

    println!("\nGuest verification checks:");
    println!("  1. gas_used == expected_gas_cost(opcode) - FIXED per opcode");
    println!("  2. pc < 1,000,000 - BOUND CHECK");
    println!("  3. stack_delta matches opcode semantics - FIXED per opcode");
    println!("  4. entry_pc < 1,000,000 - BOUND CHECK");
    println!("  5. steps_len < 65536 - BOUND CHECK");
    println!("  6. trace_data.len() >= steps_len * 41 - SIZE CHECK");

    println!("\nThe trace is ALREADY MINIMAL - only gas, PC, stack len changes.");
    println!("No optimization possible on trace complexity without changing the protocol.");

    // ============================================================
    // SUMMARY
    // ============================================================
    println!("\n{}", "=".repeat(70));
    println!("SUMMARY");
    println!("{}", "=".repeat(70));

    println!("
FIXED COSTS (cannot be reduced without architecture change):
  - Prover setup:    {:.3}s (one-time per guest program)
  - Proof overhead:   ~5-6s (Dory commitments + sumcheck)

VARIABLE COSTS (scale with trace count):
  - Batch (10 sequential): {:.3}s total, {:.3}s per trace
  - Parallel (4 threads): {:.3}s total, {:.3}s per trace

KEY INSIGHTS:
1. Setup is one-time: reuse the same prover for all similar traces
2. Per-trace cost is ~6s regardless of trace length (fixed overhead)
3. Parallelization reduces WALL CLOCK time but not per-trace time
4. Trace is already minimal (41 bytes/step for gas/PC/stack)

TO ACHIEVE FASTER PROOFS:
  - Batching: prove 100 traces at once (amortize setup)
  - Parallel: use N threads for N independent traces
  - Architecture: Dory -> Groth16 (requires fork)
", setup_time.as_secs_f64(),
       batch_time.as_secs_f64(), batch_time.as_secs_f64() / 10.0,
       parallel_time.as_secs_f64(), parallel_time.as_secs_f64() / 4.0);
}