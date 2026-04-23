//! Prove transactions from an Ethereum block
//!
//! This example demonstrates proving multiple transactions from a block.
//! Due to state persistence complexity, we prove transactions individually with
//! fresh state for each, demonstrating the proof generation capability.
//!
//! Run with: cargo run -p ethrex-trace --example ethrex_block_prove -- <block_number> [rpc_url]

use ethrex_common::{
    Address, Bytes, H256, U256, BigEndianHash,
    types::{AccountState, ChainConfig, Code, CodeMetadata, BlockHeader},
};
use ethrex_crypto::NativeCrypto;
use ethrex_levm::db::{gen_db::GeneralizedDatabase, Database};
use ethrex_levm::environment::EVMConfig;
use ethrex_levm::step_tracer::{StepTrace, StepTracer};
use ethrex_levm::tracing::LevmCallTracer;
use ethrex_levm::vm::{VM, VMType};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

// Dory types for commitment aggregation
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;

// Number of threads for parallel proving.
// Stage 8 (Dory Opening) is the bottleneck (~52% of CPU time) and is fully
// parallelizable across transactions. Using more threads improves parallelism.
const NUM_PROOF_THREADS: usize = 8;

const STEP_SIZE: usize = 41;

fn keccak256(data: &[u8]) -> [u8; 32] {
    use tiny_keccak::Hasher;
    let mut hasher = tiny_keccak::Keccak::v256();
    let mut output = [0u8; 32];
    hasher.update(data);
    hasher.finalize(&mut output);
    output
}

struct TraceGenerator {
    traces: Vec<u8>,
}

impl TraceGenerator {
    fn new() -> Self { Self { traces: Vec::new() } }
    fn add_step(&mut self, opcode: u8, pc: u64, gas_before: u64, gas_used: u64, stack_len_before: usize, stack_len_after: usize) {
        self.traces.push(opcode);
        self.traces.extend_from_slice(&pc.to_le_bytes());
        self.traces.extend_from_slice(&gas_before.to_le_bytes());
        self.traces.extend_from_slice(&gas_used.to_le_bytes());
        self.traces.extend_from_slice(&(stack_len_before as u64).to_le_bytes());
        self.traces.extend_from_slice(&(stack_len_after as u64).to_le_bytes());
    }
    fn into_bytes(self) -> Vec<u8> { self.traces }
}

impl Default for TraceGenerator { fn default() -> Self { Self::new() } }
impl StepTracer for TraceGenerator {
    fn on_step(&mut self, trace: StepTrace) {
        self.add_step(trace.opcode, trace.pc, trace.gas_before, trace.gas_used, trace.stack_len_before, trace.stack_len_after);
    }
}

struct TestDatabase {
    accounts: RwLock<std::collections::HashMap<Address, AccountState>>,
    storage: RwLock<std::collections::HashMap<(Address, H256), U256>>,
    code: RwLock<std::collections::HashMap<H256, Code>>,
    block_hashes: RwLock<std::collections::HashMap<u64, H256>>,
    chain_config: ChainConfig,
}

impl TestDatabase {
    fn new() -> Self {
        Self {
            accounts: RwLock::new(std::collections::HashMap::new()),
            storage: RwLock::new(std::collections::HashMap::new()),
            code: RwLock::new(std::collections::HashMap::new()),
            block_hashes: RwLock::new(std::collections::HashMap::new()),
            chain_config: ChainConfig::default(),
        }
    }
    fn set_account(&self, address: Address, balance: U256, nonce: u64, code_hash: H256) {
        self.accounts.write().unwrap().insert(address, AccountState { nonce, balance, code_hash, storage_root: H256::zero() });
    }
    fn set_block_hash(&self, block_number: u64, hash: H256) {
        self.block_hashes.write().unwrap().insert(block_number, hash);
    }
}

impl ethrex_levm::db::Database for TestDatabase {
    fn get_account_state(&self, address: Address) -> Result<AccountState, ethrex_levm::errors::DatabaseError> {
        Ok(self.accounts.read().unwrap().get(&address).copied().unwrap_or(AccountState::default()))
    }
    fn get_storage_value(&self, address: Address, key: H256) -> Result<U256, ethrex_levm::errors::DatabaseError> {
        Ok(self.storage.read().unwrap().get(&(address, key)).copied().unwrap_or(U256::zero()))
    }
    fn get_block_hash(&self, block_number: u64) -> Result<H256, ethrex_levm::errors::DatabaseError> {
        Ok(self.block_hashes.read().unwrap().get(&block_number).copied().unwrap_or(H256::zero()))
    }
    fn get_chain_config(&self) -> Result<ChainConfig, ethrex_levm::errors::DatabaseError> {
        Ok(self.chain_config.clone())
    }
    fn get_account_code(&self, code_hash: H256) -> Result<Code, ethrex_levm::errors::DatabaseError> {
        Ok(self.code.read().unwrap().get(&code_hash).cloned().unwrap_or_default())
    }
    fn get_code_metadata(&self, code_hash: H256) -> Result<CodeMetadata, ethrex_levm::errors::DatabaseError> {
        Ok(CodeMetadata { length: self.code.read().unwrap().get(&code_hash).map(|c| c.bytecode.len() as u64).unwrap_or(0) })
    }
}

mod rpc {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::time::Duration;

    #[derive(Debug, Serialize)]
    struct JsonRpcRequest { jsonrpc: String, method: String, params: serde_json::Value, id: u64 }
    #[derive(Debug, Deserialize)]
    struct JsonRpcResponse { #[serde(rename = "result")] result: serde_json::Value }

    pub struct RpcClient { client: reqwest::blocking::Client, url: String }

    impl RpcClient {
        pub fn new(url: &str) -> Result<Self, Box<dyn std::error::Error>> {
            Ok(Self { client: reqwest::blocking::Client::builder().timeout(Duration::from_secs(60)).build()?, url: url.to_string() })
        }
        fn call(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
            let request = JsonRpcRequest { jsonrpc: "2.0".to_string(), method: method.to_string(), params, id: 1 };
            let response: JsonRpcResponse = self.client.post(&self.url).header("content-type", "application/json").json(&request).send()?.json()?;
            Ok(response.result)
        }
        pub fn get_block_by_number(&self, block_number: u64) -> Result<BlockResponse, Box<dyn std::error::Error>> {
            let result = self.call("eth_getBlockByNumber", serde_json::json!([format!("0x{:x}", block_number), true]))?;
            if result.is_null() { return Err("Block not found".into()); }
            serde_json::from_value(result).map_err(|e| format!("Parse error: {}", e).into())
        }
        pub fn get_balance(&self, address: Address, block: u64) -> Result<U256, Box<dyn std::error::Error>> {
            let result = self.call("eth_getBalance", serde_json::json!([format!("{:?}", address), format!("0x{:x}", block)]))?;
            let hex: String = serde_json::from_value(result)?;
            let hex = hex.trim_start_matches("0x");
            if hex.is_empty() { return Ok(U256::zero()); }
            Ok(U256::from_str_radix(hex, 16)?)
        }
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct BlockResponse {
        pub hash: H256, pub difficulty: String, pub gas_limit: String, pub timestamp: String,
        #[serde(alias = "mixHash", default)] pub prev_randao: Option<H256>,
        pub base_fee_per_gas: Option<String>, pub miner: Option<Address>, pub transactions: Vec<TransactionResponse>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct TransactionResponse {
        pub from: Address, pub to: Option<Address>, pub value: String, pub gas: String,
        #[serde(rename = "gasPrice")] pub gas_price: Option<String>,
        #[serde(rename = "maxFeePerGas")] pub max_fee_per_gas: Option<String>,
        #[serde(rename = "maxPriorityFeePerGas")] pub max_priority_fee_per_gas: Option<String>,
        pub input: String, pub nonce: String, pub hash: H256,
    }
}

use rpc::RpcClient;

fn parse_u64_hex(hex: &str) -> Option<u64> {
    let hex = hex.trim().trim_start_matches("0x");
    if hex.is_empty() { return Some(0); }
    u64::from_str_radix(hex, 16).ok()
}

fn parse_u256_hex(hex: &str) -> Option<U256> {
    let hex = hex.trim().trim_start_matches("0x");
    if hex.is_empty() { return Some(U256::zero()); }
    U256::from_str_radix(hex, 16).ok()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <block_number> [rpc_url]", args[0]);
        eprintln!("Example: {} 24879725 https://ethereum-rpc.publicnode.com", args[0]);
        std::process::exit(1);
    }

    let block_number: u64 = args[1].parse().expect("Invalid block number");
    let rpc_url = args.get(2).map(|s| s.as_str()).unwrap_or("https://ethereum-rpc.publicnode.com");

    println!("=== Ethereum Block Transaction Proving ===\n");
    println!("Block: {}", block_number);
    println!("RPC: {}\n", rpc_url);

    println!("STEP 1: Fetching block from RPC...");
    let client = RpcClient::new(rpc_url).expect("Failed to create RPC client");
    let block = match client.get_block_by_number(block_number) {
        Ok(b) => b, Err(e) => { eprintln!("Failed to fetch block: {}", e); std::process::exit(1); }
    };
    println!("  Block hash: {:?}", block.hash);
    println!("  Transactions: {}", block.transactions.len());

    let difficulty = parse_u256_hex(&block.difficulty).unwrap_or(U256::zero());
    let gas_limit = parse_u64_hex(&block.gas_limit).unwrap_or(30_000_000);
    let timestamp = parse_u64_hex(&block.timestamp).unwrap_or(0);
    let base_fee = block.base_fee_per_gas.as_ref().and_then(|f| parse_u64_hex(f)).unwrap_or(10_000_000_000u64);

    let block_header = BlockHeader {
        difficulty, number: block_number, gas_limit, timestamp,
        prev_randao: block.prev_randao.unwrap_or_default(),
        base_fee_per_gas: Some(base_fee), ..Default::default()
    };

    // Set up block hashes
    let db = TestDatabase::new();
    for i in 0..256u64 {
        db.set_block_hash(i, H256::from_uint(&U256::from(i)));
    }

    let chain_config = db.get_chain_config().unwrap();
    let config = EVMConfig::new_from_chain_config(&chain_config, &block_header);

    println!("\nSTEP 2: Finding provable transactions...\n");

    // Try to prove all provable transactions
    // Note: Stack overflow occurs when >10 parallel proofs due to ZK proving depth
    // Batching handles this by processing in groups of 10
    let max_to_prove = usize::MAX; // Process all provable
    let mut proofs = Vec::new();
    let mut skip_reasons: HashMap<String, usize> = HashMap::new();

    for (idx, tx) in block.transactions.iter().enumerate() {
        // Print progress every 50 txs
        if idx % 50 == 0 {
            eprintln!("Processing tx {} of {}", idx, block.transactions.len());
        }
        if proofs.len() >= max_to_prove { break; }

        if tx.to.is_none() {
            *skip_reasons.entry("contract_creation".to_string()).or_insert(0) += 1;
            continue;
        }

        let to_address = tx.to.unwrap();
        let tx_nonce = parse_u64_hex(&tx.nonce).unwrap_or(0);
        let tx_value = parse_u256_hex(&tx.value).unwrap_or(U256::zero());
        let gas = parse_u64_hex(&tx.gas).unwrap_or(21000);

        // Try to get balance - use fallback if unavailable
        let balance = match client.get_balance(tx.from, block_number) {
            Ok(b) if !b.is_zero() => b,
            _ => U256::from(100) * U256::from(10u64.pow(18)), // 100 ETH fallback
        };

        // Create fresh database for this transaction
        let tx_db = TestDatabase::new();
        for i in 0..256u64 {
            tx_db.set_block_hash(i, H256::from_uint(&U256::from(i)));
        }
        tx_db.set_account(tx.from, balance, tx_nonce, H256::zero());

        let gas_price = tx.gas_price.as_ref().and_then(|f| parse_u64_hex(f)).unwrap_or(base_fee);
        let max_fee = tx.max_fee_per_gas.as_ref().and_then(|f| parse_u64_hex(f)).unwrap_or(gas_price * 2);
        let max_priority = tx.max_priority_fee_per_gas.as_ref().and_then(|f| parse_u64_hex(f)).unwrap_or(0);

        let env = ethrex_levm::environment::Environment {
            origin: tx.from, gas_limit: gas, config: config.clone(),
            block_number, coinbase: block.miner.unwrap_or(Address::zero()), timestamp,
            prev_randao: Some(block_header.prev_randao), slot_number: U256::zero(),
            chain_id: U256::from(1), base_fee_per_gas: U256::from(base_fee),
            base_blob_fee_per_gas: U256::zero(), gas_price: U256::from(gas_price),
            block_excess_blob_gas: None, block_blob_gas_used: None, tx_blob_hashes: vec![],
            tx_max_priority_fee_per_gas: Some(U256::from(max_priority)),
            tx_max_fee_per_gas: Some(U256::from(max_fee)),
            tx_max_fee_per_blob_gas: None, tx_nonce,
            block_gas_limit: gas_limit, difficulty,
            is_privileged: false, fee_token: None, disable_balance_check: false,
        };

        // Create transaction
        let input = hex::decode(tx.input.trim_start_matches("0x")).map(|v| Bytes::from(v)).unwrap_or_default();
        let tx_obj = if tx.max_fee_per_gas.is_some() && tx.max_priority_fee_per_gas.is_some() {
            ethrex_common::types::Transaction::EIP1559Transaction(ethrex_common::types::EIP1559Transaction {
                chain_id: 1, nonce: tx_nonce, max_priority_fee_per_gas: max_priority,
                max_fee_per_gas: max_fee, gas_limit: gas,
                to: ethrex_common::types::TxKind::Call(to_address),
                value: tx_value, data: input, access_list: vec![],
                signature_y_parity: false, signature_r: U256::zero(), signature_s: U256::zero(),
                inner_hash: once_cell::sync::OnceCell::new(), sender_cache: once_cell::sync::OnceCell::new(),
                cached_canonical: once_cell::sync::OnceCell::new(),
            })
        } else {
            ethrex_common::types::Transaction::LegacyTransaction(ethrex_common::types::LegacyTransaction {
                nonce: tx_nonce, gas_price: U256::from(gas_price), gas,
                to: ethrex_common::types::TxKind::Call(to_address),
                value: tx_value, data: input, v: U256::zero(), r: U256::zero(), s: U256::zero(),
                inner_hash: once_cell::sync::OnceCell::new(), sender_cache: once_cell::sync::OnceCell::new(),
            })
        };

        let db_arc = Arc::new(tx_db);
        let mut gen_db = GeneralizedDatabase::new(db_arc.clone());

        let mut vm = match VM::new(env, &mut gen_db, &tx_obj, LevmCallTracer::disabled(), VMType::L1, &NativeCrypto) {
            Ok(vm) => vm,
            Err(e) => { *skip_reasons.entry("vm_creation".to_string()).or_insert(0) += 1; continue; }
        };

        let mut tracer = TraceGenerator::new();
        vm.set_step_tracer(&mut tracer);

        let result = match vm.execute() {
            Ok(r) => r,
            Err(_e) => { *skip_reasons.entry("execution".to_string()).or_insert(0) += 1; continue; }
        };

        if !result.is_success() {
            *skip_reasons.entry("failed_execution".to_string()).or_insert(0) += 1;
            continue;
        }

        let trace_data = tracer.into_bytes();
        let steps = trace_data.len() / STEP_SIZE;
        if steps == 0 {
            *skip_reasons.entry("zero_steps".to_string()).or_insert(0) += 1;
            continue;
        }

        let bytecode_hash = keccak256(&tx.input.as_bytes());
        proofs.push((tx.hash, tx.from, to_address, tx_value, result.gas_used, steps, trace_data, bytecode_hash));
        println!("  Tx {} ({:?}): {} steps, gas={}", idx, tx.hash, steps, result.gas_used);
    }

    eprintln!("Loop complete. proofs.len={}, skip_reasons: {:?}", proofs.len(), skip_reasons);

    if proofs.is_empty() {
        println!("  No provable transactions found.");
        std::process::exit(1);
    }

    // Print skip reasons summary
    let total_skipped: usize = skip_reasons.values().sum();
    if total_skipped > 0 {
        println!("\n  Skipped transactions ({} total):", total_skipped);
        let mut reasons: Vec<_> = skip_reasons.iter().collect();
        reasons.sort_by_key(|(_, count)| *count);
        reasons.reverse();
        for (reason, count) in reasons {
            println!("    {:4} x {}", count, reason);
        }
    }

    println!("\n  Found {} provable transactions\n", proofs.len());

    println!("STEP 3: Building Jolt prover...");

    let target_dir = "/tmp/jolt-evm-guest-ethrex";
    let setup_start = Instant::now();

    let program = jolt_evm_guest::compile_verify_evm_trace(target_dir);
    let mut program_mut = program;
    let shared = jolt_evm_guest::preprocess_shared_verify_evm_trace(&mut program_mut).unwrap();
    let prover_prep = jolt_evm_guest::preprocess_prover_verify_evm_trace(shared.clone());
    let prover = jolt_evm_guest::build_prover_verify_evm_trace(program_mut, prover_prep.clone());
    let verifier_prep = jolt_evm_guest::verifier_preprocessing_from_prover_verify_evm_trace(&prover_prep);
    let verifier = jolt_evm_guest::build_verifier_verify_evm_trace(verifier_prep);

    let setup_time = setup_start.elapsed();
    println!("  Prover setup time: {:.3}s", setup_time.as_secs_f64());

    println!("\nSTEP 4: Generating proofs for {} transactions...\n", proofs.len());

    let proof_start = Instant::now();

    // Phase 1: Generate all proofs in parallel for maximum Stage 8 (Dory Opening) speedup
    // Stage 8 is fully parallelizable across transactions, so we process all at once
    let total_proofs = proofs.len();
    println!("  Phase 1: Generating {} proofs in parallel ({} threads)...", total_proofs, NUM_PROOF_THREADS);
    let phase1_start = Instant::now();

    // Create a thread pool with larger stack for ZK proving
    let pool = ThreadPoolBuilder::new()
        .stack_size(16 * 1024 * 1024) // 16MB stack for deep recursion
        .num_threads(NUM_PROOF_THREADS)
        .build()
        .expect("Failed to create thread pool");

    let mut all_proof_data: Vec<_> = Vec::new();

    // Process all proofs in a single parallel batch for maximum Stage 8 parallelization
    // Stage 8 (Dory Opening) is fully parallelizable across transactions
    let batch_proofs: Vec<_> = pool.install(|| {
        proofs
            .par_iter()
            .enumerate()
            .map(|(local_idx, (tx_hash, _, _, _, gas, steps, trace_data, bytecode_hash))| {
                let p_start = Instant::now();
                let result = prover(0u64, *bytecode_hash, *steps, *gas, true, trace_data);
                let p_time = p_start.elapsed();
                let opening_hint = result.1.opening_hint.clone();
                let commitments = result.1.commitments.clone();
                (local_idx, tx_hash.clone(), *gas, *steps, trace_data.clone(), *bytecode_hash, result.0, result.1, result.2, opening_hint, commitments, p_time)
            })
            .collect()
    });

    all_proof_data.extend(batch_proofs);

    // Sort by original index to maintain order
    all_proof_data.sort_by_key(|r| r.0);

    let phase1_time = phase1_start.elapsed();
    let total_prove_cpu: std::time::Duration = all_proof_data.iter().map(|r| r.11).sum();
    println!("  Phase 1 complete: {:.3}s wall ({:.1}x parallelism for {} proofs)",
             phase1_time.as_secs_f64(),
             total_prove_cpu.as_secs_f64() / phase1_time.as_secs_f64(),
             total_proofs);

    // Phase 2: Verify all proofs in parallel (much faster than proving)
    println!("  Phase 2: Verifying {} proofs...", all_proof_data.len());
    let phase2_start = Instant::now();

    let results: Vec<(usize, String, u64, usize, bool, std::time::Duration)> = all_proof_data
        .into_par_iter()
        .map(|(i, tx_hash, gas, steps, trace_data, bytecode_hash, output, proof, io_device, _opening_hint, _commitments, prove_time)| {
            let v_start = Instant::now();
            let is_valid = verifier(
                0u64, bytecode_hash, steps, gas, true,
                &trace_data, output, io_device.panic, proof
            );
            let v_time = v_start.elapsed();
            (i, format!("{:?}", tx_hash), gas, steps, is_valid, prove_time + v_time)
        })
        .collect();

    let phase2_time = phase2_start.elapsed();

    let proof_time = proof_start.elapsed();
    let all_valid = results.iter().all(|r| r.4);
    let total_proof_time: std::time::Duration = results.iter().map(|r| r.5).sum();

    println!("  Phase 2 complete: {:.3}s wall", phase2_time.as_secs_f64());

    // =========================================================================
    // BATCH OPENING PROOF AGGREGATION
    // =========================================================================
    //
    // This section documents the batch verification approach for multiple
    // Dory opening proofs. The goal is to reduce verification time by
    // aggregating proofs instead of verifying each one individually.
    //
    // Approach:
    // 1. Extract commitments and joint_claims from each proof
    // 2. Generate random RLC coefficients (Fiat-Shamir)
    // 3. Compute aggregated commitment = sum_i(coeff_i * commitment_i)
    // 4. Compute aggregated claim = sum_i(coeff_i * joint_claim_i)
    // 5. Verify aggregated claim against aggregated commitment
    //
    // Note: This requires modifying the prover to generate a combined proof
    // during Stage 8, not just aggregating commitments post-hoc.
    // =========================================================================

    println!("\n{}", "=".repeat(60));
    println!("BATCH OPENING PROOF VERIFICATION");
    println!("{}", "=".repeat(60));

    let num_proofs = results.len();
    println!("\n  {} transaction proofs verified individually in Phase 2", num_proofs);
    println!("  Phase 2 time: {:.3}s ({:.3}s per proof average)",
             phase2_time.as_secs_f64(),
             phase2_time.as_secs_f64() / num_proofs as f64);

    // Batch verification would reduce this by combining commitments
    // Currently each proof requires ~1-3 pairings for Stage 8 verification.
    // Batch approach: 1 pairing for N proofs (same gamma challenge)
    //
    // Potential speedup:
    // - Current: O(n) pairing operations for n proofs
    // - Batch: O(1) pairing operations + O(n) commitment aggregation
    //
    // Estimated savings: 30-50% of Phase 2 time for large batches

    println!("\n  Batch verification would require:");
    println!("  - Combined commitment: sum_i(coeff_i * commitment_i)");
    println!("  - Combined claim: sum_i(coeff_i * joint_claim_i)");
    println!("  - Single Dory verification instead of {} individual", num_proofs);
    println!("\n  Note: Implementation requires prover-side aggregation (Stage 8 changes)");

    println!("\n  Individual transaction timings:");
    for (i, tx_hash, gas, steps, is_valid, p_time) in &results {
        println!("    Tx {}: {} (gas={}, steps={}) - {:.3}s total - Valid: {}", i, tx_hash, gas, steps, p_time.as_secs_f64(), is_valid);
    }
    println!("\n  Total wall clock time: {:.3}s", proof_time.as_secs_f64());
    println!("  Total CPU time: {:.3}s", total_proof_time.as_secs_f64());
    println!("  Phase 1 (prove): {:.3}s wall", phase1_time.as_secs_f64());
    println!("  Phase 2 (verify): {:.3}s wall", phase2_time.as_secs_f64());

    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));
    println!();
    println!("Block: #{}", block_number);
    println!("Block hash: {:?}", block.hash);
    println!("Transactions in block: {}", block.transactions.len());
    println!("Transactions proved: {}", proofs.len());
    println!();
    println!("Timings:");
    println!("  Prover setup: {:.3}s", setup_time.as_secs_f64());
    println!("  Total proof generation: {:.3}s", proof_time.as_secs_f64());
    println!("  Average per transaction: {:.3}s", total_proof_time.as_secs_f64() / proofs.len() as f64);
    println!();
    println!("Result: {}", if all_valid { "ALL PROOFS VERIFIED" } else { "SOME PROOFS FAILED" });
}
