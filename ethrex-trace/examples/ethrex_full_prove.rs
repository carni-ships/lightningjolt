//! Full EVM Execution and Jolt Proof Generation
//!
//! This example demonstrates the complete pipeline:
//! 1. Execute a transaction on ethrex VM with step tracing
//! 2. Generate 41-byte/step trace data
//! 3. Feed to Jolt prover and generate proof
//! 4. Verify the proof
//!
//! Run with: cargo run -p ethrex-trace --example ethrex_full_prove

use ethrex_common::{
    Address, Bytes, H256, U256,
    types::{AccountState, ChainConfig, Code, CodeMetadata, TxKind},
};
use ethrex_crypto::NativeCrypto;
use ethrex_levm::db::gen_db::GeneralizedDatabase;
use ethrex_levm::environment::EVMConfig;
use ethrex_levm::step_tracer::{StepTrace, StepTracer};
use ethrex_levm::tracing::LevmCallTracer;
use ethrex_levm::vm::{VM, VMType};
use std::str::FromStr;
use std::sync::{Arc, RwLock};
use std::time::Instant;

// Constants from jolt-evm for trace format compatibility
const STEP_SIZE: usize = 41;

/// Keccak256 hash function
fn keccak256(data: &[u8]) -> [u8; 32] {
    use tiny_keccak::Hasher;
    let mut hasher = tiny_keccak::Keccak::v256();
    let mut output = [0u8; 32];
    hasher.update(data);
    hasher.finalize(&mut output);
    output
}

/// Trace generator that produces 41-byte step format for Jolt
struct TraceGenerator {
    traces: Vec<u8>,
}

impl TraceGenerator {
    fn new() -> Self {
        Self { traces: Vec::new() }
    }

    fn add_step(&mut self, opcode: u8, pc: u64, gas_before: u64, gas_used: u64, stack_len_before: usize, stack_len_after: usize) {
        self.traces.push(opcode);
        self.traces.extend_from_slice(&pc.to_le_bytes());
        self.traces.extend_from_slice(&gas_before.to_le_bytes());
        self.traces.extend_from_slice(&gas_used.to_le_bytes());
        self.traces.extend_from_slice(&(stack_len_before as u64).to_le_bytes());
        self.traces.extend_from_slice(&(stack_len_after as u64).to_le_bytes());
    }

    fn into_bytes(self) -> Vec<u8> {
        self.traces
    }
}

impl Default for TraceGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl StepTracer for TraceGenerator {
    fn on_step(&mut self, trace: StepTrace) {
        self.add_step(
            trace.opcode,
            trace.pc,
            trace.gas_before,
            trace.gas_used,
            trace.stack_len_before,
            trace.stack_len_after,
        );
    }
}

/// In-memory database with pre-set state for EVM execution
struct RealisticDatabase {
    accounts: RwLock<std::collections::HashMap<Address, AccountState>>,
    storage: RwLock<std::collections::HashMap<(Address, H256), U256>>,
    code: RwLock<std::collections::HashMap<H256, Code>>,
    code_metadata: RwLock<std::collections::HashMap<H256, CodeMetadata>>,
    block_hashes: RwLock<std::collections::HashMap<u64, H256>>,
    chain_config: ChainConfig,
}

impl Default for RealisticDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl RealisticDatabase {
    fn new() -> Self {
        Self {
            accounts: RwLock::new(std::collections::HashMap::new()),
            storage: RwLock::new(std::collections::HashMap::new()),
            code: RwLock::new(std::collections::HashMap::new()),
            code_metadata: RwLock::new(std::collections::HashMap::new()),
            block_hashes: RwLock::new(std::collections::HashMap::new()),
            chain_config: ChainConfig::default(),
        }
    }

    fn setup_storage_contract(&self, address: Address, bytecode: Bytes) -> H256 {
        let code_obj = Code::from_bytecode(bytecode.clone(), &NativeCrypto);
        let code_hash = code_obj.hash;
        self.code.write().unwrap().insert(code_hash, code_obj);
        self.code_metadata.write().unwrap().insert(code_hash, CodeMetadata { length: bytecode.len() as u64 });

        let mut accounts = self.accounts.write().unwrap();
        accounts.insert(address, AccountState {
            nonce: 1,
            balance: U256::from(1_000_000_000_000_000_000u64),
            code_hash,
            storage_root: H256::zero(),
        });
        code_hash
    }

    fn setup_account(&self, address: Address, balance: U256, nonce: u64) {
        let mut accounts = self.accounts.write().unwrap();
        accounts.insert(address, AccountState {
            nonce,
            balance,
            code_hash: H256::zero(),
            storage_root: H256::zero(),
        });
    }

    fn set_storage(&self, address: Address, slot: H256, value: U256) {
        let mut storage = self.storage.write().unwrap();
        storage.insert((address, slot), value);
    }

    fn set_block_hash(&self, block_number: u64, hash: H256) {
        let mut hashes = self.block_hashes.write().unwrap();
        hashes.insert(block_number, hash);
    }
}

impl ethrex_levm::db::Database for RealisticDatabase {
    fn get_account_state(&self, address: Address) -> Result<AccountState, ethrex_levm::errors::DatabaseError> {
        let accounts = self.accounts.read().unwrap();
        Ok(accounts.get(&address).copied().unwrap_or(AccountState::default()))
    }

    fn get_storage_value(&self, address: Address, key: H256) -> Result<U256, ethrex_levm::errors::DatabaseError> {
        let storage = self.storage.read().unwrap();
        Ok(storage.get(&(address, key)).copied().unwrap_or(U256::zero()))
    }

    fn get_block_hash(&self, block_number: u64) -> Result<H256, ethrex_levm::errors::DatabaseError> {
        let hashes = self.block_hashes.read().unwrap();
        Ok(hashes.get(&block_number).copied().unwrap_or(H256::zero()))
    }

    fn get_chain_config(&self) -> Result<ChainConfig, ethrex_levm::errors::DatabaseError> {
        Ok(self.chain_config)
    }

    fn get_account_code(&self, code_hash: H256) -> Result<Code, ethrex_levm::errors::DatabaseError> {
        let code = self.code.read().unwrap();
        Ok(code.get(&code_hash).cloned().unwrap_or_default())
    }

    fn get_code_metadata(&self, code_hash: H256) -> Result<CodeMetadata, ethrex_levm::errors::DatabaseError> {
        let metadata = self.code_metadata.read().unwrap();
        Ok(metadata.get(&code_hash).copied().unwrap_or(CodeMetadata { length: 0 }))
    }
}

/// Create an EIP-1559 transaction
fn create_call_tx(
    from: Address,
    to: Address,
    value: U256,
    gas: u64,
    input: Bytes,
) -> ethrex_common::types::Transaction {
    let tx = ethrex_common::types::EIP1559Transaction {
        chain_id: 1,
        nonce: 0,
        max_priority_fee_per_gas: 1_000_000_000,
        max_fee_per_gas: 100_000_000_000,
        gas_limit: gas,
        to: TxKind::Call(to),
        value,
        data: input,
        access_list: vec![],
        signature_y_parity: false,
        signature_r: U256::zero(),
        signature_s: U256::zero(),
        inner_hash: Default::default(),
        sender_cache: Default::default(),
        cached_canonical: Default::default(),
    };
    ethrex_common::types::Transaction::EIP1559Transaction(tx)
}

fn main() {
    println!("=== Full EVM Execution and Jolt Proof Generation ===\n");

    // ============================================================
    // STEP 1: Execute transaction on ethrex VM
    // ============================================================
    println!("STEP 1: Executing transaction on ethrex VM...\n");

    let db = RealisticDatabase::new();

    // Set up sender account
    let sender = Address::from_str("0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266").unwrap();
    db.setup_account(sender, U256::from(10u64.pow(18)), 0);

    // Set up a simple storage contract
    // bytecode: PUSH1 5 PUSH1 3 ADD STOP - simple addition
    let contract_bytecode = Bytes::from(vec![
        0x60, 0x05, // PUSH1 5
        0x60, 0x03, // PUSH1 3
        0x01,       // ADD
        0x00,       // STOP
    ]);
    let contract = Address::from_str("0x1234567890123456789012345678901234567890").unwrap();
    let _contract_code_hash = db.setup_storage_contract(contract, contract_bytecode.clone());

    // Set block hashes
    for i in 0..256u64 {
        let hash = H256::from_str(&format!("0x{:064x}", i)).unwrap();
        db.set_block_hash(i, hash);
    }

    // Create transaction: simple call that just executes bytecode (no function call)
    // Input is empty - just executes the bytecode directly
    let input = Bytes::new();

    let tx = create_call_tx(sender, contract, U256::zero(), 1_000_000, input);

    println!("  From: {}", sender);
    println!("  To: {}", contract);
    println!("  Data: store(99)");
    println!();

    // Build block header
    let block_header = ethrex_common::types::BlockHeader {
        difficulty: U256::from(2u64.pow(31)),
        number: 1,
        gas_limit: 30_000_000,
        timestamp: 1_600_000_000,
        prev_randao: H256::zero(),
        base_fee_per_gas: Some(10u64.pow(9)),
        ..Default::default()
    };

    // Wrap database and build environment
    let db = Arc::new(db);
    let mut gen_db = GeneralizedDatabase::new(db);

    let chain_config = gen_db.store.get_chain_config().unwrap();
    let config = EVMConfig::new_from_chain_config(&chain_config, &block_header);

    let env = ethrex_levm::environment::Environment {
        origin: sender,
        gas_limit: tx.gas_limit(),
        config,
        block_number: block_header.number,
        coinbase: Address::zero(),
        timestamp: block_header.timestamp,
        prev_randao: Some(block_header.prev_randao),
        slot_number: U256::zero(),
        chain_id: U256::from(1),
        base_fee_per_gas: U256::from(block_header.base_fee_per_gas.unwrap_or(10u64.pow(9))),
        base_blob_fee_per_gas: U256::zero(),
        gas_price: tx.gas_price(),
        block_excess_blob_gas: None,
        block_blob_gas_used: None,
        tx_blob_hashes: vec![],
        tx_max_priority_fee_per_gas: tx.max_priority_fee().map(U256::from),
        tx_max_fee_per_gas: tx.max_fee_per_gas().map(U256::from),
        tx_max_fee_per_blob_gas: tx.max_fee_per_blob_gas(),
        tx_nonce: tx.nonce(),
        block_gas_limit: block_header.gas_limit,
        difficulty: block_header.difficulty,
        is_privileged: false,
        fee_token: None,
        disable_balance_check: false,
    };

    // Create VM and execute
    let mut vm = VM::new(
        env,
        &mut gen_db,
        &tx,
        LevmCallTracer::disabled(),
        VMType::L1,
        &NativeCrypto,
    ).unwrap();

    let mut tracer = TraceGenerator::new();
    vm.set_step_tracer(&mut tracer);

    let exec_start = Instant::now();
    let result = vm.execute().unwrap();
    let exec_time = exec_start.elapsed();

    let trace_data = tracer.into_bytes();
    let steps_len = trace_data.len() / STEP_SIZE;
    let final_gas = result.gas_used;
    let success = result.is_success();

    println!("  Execution completed in {:?}", exec_time);
    println!("  Success: {}", success);
    println!("  Gas used: {}", final_gas);
    println!("  Trace: {} steps, {} bytes", steps_len, trace_data.len());

    // ============================================================
    // STEP 2: Build Jolt prover
    // ============================================================
    println!("\nSTEP 2: Building Jolt prover...\n");

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

    // ============================================================
    // STEP 3: Generate proof
    // ============================================================
    println!("\nSTEP 3: Generating Jolt proof...\n");

    // Compute bytecode hash
    let bytecode_hash = keccak256(&contract_bytecode);
    let entry_pc = 0u64;

    let proof_start = Instant::now();
    let (output, proof, io_device) = prover(
        entry_pc,
        bytecode_hash,
        steps_len,
        final_gas,
        success,
        &trace_data,
    );
    let proof_time = proof_start.elapsed();

    println!("  Proof generated in {:.3}s", proof_time.as_secs_f64());

    // ============================================================
    // STEP 4: Verify proof
    // ============================================================
    println!("\nSTEP 4: Verifying proof...\n");

    let is_valid = verifier(
        entry_pc,
        bytecode_hash,
        steps_len,
        final_gas,
        success,
        &trace_data,
        output,
        io_device.panic,
        proof,
    );

    println!("  Verification result: {}", is_valid);

    // ============================================================
    // SUMMARY
    // ============================================================
    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));
    println!();
    println!("Transaction:");
    println!("  Contract bytecode: {} bytes", contract_bytecode.len());
    println!("  Entry PC: {}", entry_pc);
    println!("  Steps: {}", steps_len);
    println!("  Final gas: {}", final_gas);
    println!("  Success: {}", success);
    println!();
    println!("Timings:");
    println!("  VM execution: {:?}", exec_time);
    println!("  Prover setup: {:.3}s", setup_time.as_secs_f64());
    println!("  Proof generation: {:.3}s", proof_time.as_secs_f64());
    println!();
    println!("Result: {}", if is_valid { "PROOF VERIFIED ✓" } else { "VERIFICATION FAILED ✗" });
}
