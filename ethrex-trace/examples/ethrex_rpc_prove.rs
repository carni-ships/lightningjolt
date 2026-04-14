//! Fetch a real Ethereum block via RPC and generate a Jolt proof
//!
//! This example demonstrates:
//! 1. Connecting to an Ethereum RPC endpoint (Infura/Alchemy/local)
//! 2. Fetching a specific block by number
//! 3. Extracting transaction data and building EVM state
//! 4. Executing a transaction through ethrex VM with tracing
//! 5. Generating and verifying a Jolt proof
//!
//! Run with: cargo run -p ethrex-trace --example ethrex_rpc_prove -- <block_number> [rpc_url]
//!
//! Example:
//!   cargo run -p ethrex-trace --example ethrex_rpc_prove -- 20000000
//!   cargo run -p ethrex-trace --example ethrex_rpc_prove -- 20000000 https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY

use ethrex_common::{
    Address, Bytes, H256, U256, BigEndianHash,
    types::{AccountState, ChainConfig, Code, CodeMetadata, BlockHeader},
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

// Constants
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

    fn setup_account(&self, address: Address, balance: U256, nonce: u64, code_hash: H256) {
        let mut accounts = self.accounts.write().unwrap();
        accounts.insert(address, AccountState {
            nonce,
            balance,
            code_hash,
            storage_root: H256::zero(),
        });
    }

    fn setup_contract(&self, address: Address, bytecode: Bytes) -> H256 {
        let code_obj = Code::from_bytecode(bytecode.clone(), &NativeCrypto);
        let code_hash = code_obj.hash;
        self.code.write().unwrap().insert(code_hash, code_obj);
        self.code_metadata.write().unwrap().insert(code_hash, CodeMetadata { length: bytecode.len() as u64 });
        code_hash
    }

    fn set_storage(&self, address: Address, slot: H256, value: U256) {
        let mut storage = self.storage.write().unwrap();
        storage.insert((address, slot), value);
    }

    fn set_block_hash(&self, block_number: u64, hash: H256) {
        let mut hashes = self.block_hashes.write().unwrap();
        hashes.insert(block_number, hash);
    }

    fn get_account_state(&self, address: Address) -> AccountState {
        self.accounts.read().unwrap().get(&address).copied().unwrap_or(AccountState::default())
    }

    fn get_code(&self, code_hash: H256) -> Code {
        self.code.read().unwrap().get(&code_hash).cloned().unwrap_or_default()
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

/// Simple synchronous RPC client using reqwest::blocking
mod rpc {
    use ethrex_common::{Address, H256, U256};
    use serde::{Deserialize, Serialize};
    use std::time::Duration;

    #[derive(Debug, Serialize)]
    struct JsonRpcRequest {
        jsonrpc: String,
        method: String,
        params: serde_json::Value,
        id: u64,
    }

    #[derive(Debug, Deserialize)]
    struct JsonRpcResponse {
        #[serde(rename = "id")]
        _id: u64,
        #[serde(rename = "jsonrpc")]
        _jsonrpc: String,
        result: serde_json::Value,
    }

    pub struct RpcClient {
        client: reqwest::blocking::Client,
        url: String,
    }

    impl RpcClient {
        pub fn new(url: &str) -> Result<Self, Box<dyn std::error::Error>> {
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()?;
            Ok(Self {
                client,
                url: url.to_string(),
            })
        }

        fn call(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
            let request = JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                method: method.to_string(),
                params,
                id: 1,
            };

            // Debug: print the request
            eprintln!("DEBUG: Calling {} with params {}", method, request.params);

            let response = self.client
                .post(&self.url)
                .header("content-type", "application/json")
                .json(&request)
                .send()?;

            let json: serde_json::Value = response.json()?;

            // Debug: print the response result status
            eprintln!("DEBUG: Response result is null: {}", json.get("result").map_or(false, |v| v.is_null()));

            if let Some(error) = json.get("error") {
                return Err(format!("RPC error: {}", error).into());
            }

            let response: JsonRpcResponse = serde_json::from_value(json)?;
            Ok(response.result)
        }

        /// Fetch block by number, returns (hash, parent_hash, state_root, timestamp, number, gas_limit, difficulty, transactions)
        pub fn get_block_by_number(&self, block_number: u64) -> Result<BlockResponse, Box<dyn std::error::Error>> {
            let params = serde_json::json!([
                format!("0x{:x}", block_number),
                true  // full transactions
            ]);
            let result = self.call("eth_getBlockByNumber", params)?;

            // Check if result is null (block not found)
            if result.is_null() {
                return Err(format!("Block {} not found (RPC returned null)", block_number).into());
            }

            // Clone result for error message since we need to borrow it
            let result_clone = result.clone();
            serde_json::from_value(result).map_err(|e| {
                format!("Failed to parse block response: {}. Response snippet: {}", e, &result_clone.to_string()[..result_clone.to_string().len().min(200)]).into()
            })
        }

        /// Fetch account balance
        pub fn get_balance(&self, address: Address, block_number: u64) -> Result<U256, Box<dyn std::error::Error>> {
            let params = serde_json::json!([
                format!("{:?}", address),
                format!("0x{:x}", block_number)
            ]);
            let result = self.call("eth_getBalance", params)?;
            let hex_str: String = serde_json::from_value(result)?;
            parse_hex_u256(&hex_str)
        }

        /// Fetch account nonce
        pub fn get_nonce(&self, address: Address, block_number: u64) -> Result<u64, Box<dyn std::error::Error>> {
            let params = serde_json::json!([
                format!("{:?}", address),
                format!("0x{:x}", block_number)
            ]);
            let result = self.call("eth_getTransactionCount", params)?;
            let hex_str: String = serde_json::from_value(result)?;
            parse_hex_u64(&hex_str)
        }

        /// Fetch contract code
        pub fn get_code(&self, address: Address, block_number: u64) -> Result<ethrex_common::Bytes, Box<dyn std::error::Error>> {
            let params = serde_json::json!([
                format!("{:?}", address),
                format!("0x{:x}", block_number)
            ]);
            let result = self.call("eth_getCode", params)?;
            let hex_str: String = serde_json::from_value(result)?;
            Ok(hex::decode(hex_str.trim_start_matches("0x"))?.into())
        }

        /// Fetch storage value at a slot
        pub fn get_storage_at(&self, address: Address, slot: H256, block_number: u64) -> Result<U256, Box<dyn std::error::Error>> {
            let params = serde_json::json!([
                format!("{:?}", address),
                format!("{:?}", slot),
                format!("0x{:x}", block_number)
            ]);
            let result = self.call("eth_getStorageAt", params)?;
            let hex_str: String = serde_json::from_value(result)?;
            parse_hex_u256(&hex_str)
        }
    }

    fn parse_hex_u64(hex: &str) -> Result<u64, Box<dyn std::error::Error>> {
        let hex = hex.trim_start_matches("0x");
        Ok(u64::from_str_radix(hex, 16)?)
    }

    fn parse_hex_u256(hex: &str) -> Result<U256, Box<dyn std::error::Error>> {
        let hex = hex.trim_start_matches("0x");
        if hex.is_empty() {
            return Ok(U256::zero());
        }
        Ok(U256::from_str_radix(hex, 16)?)
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct BlockResponse {
        pub hash: H256,
        pub parent_hash: H256,
        #[serde(rename = "stateRoot")]
        pub state_root: H256,
        #[serde(rename = "receiptsRoot")]
        pub receipts_root: H256,
        #[serde(rename = "logsBloom")]
        pub logs_bloom: serde_json::Value,
        pub difficulty: String,
        pub number: String,
        pub gas_limit: String,
        pub gas_used: String,
        pub timestamp: String,
        #[serde(alias = "mixHash", default)]
        pub prev_randao: Option<H256>,
        pub base_fee_per_gas: Option<String>,
        pub transactions: Vec<TransactionResponse>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct TransactionResponse {
        #[serde(rename = "type")]
        pub tx_type: String,
        pub from: Address,
        pub to: Option<Address>,
        pub value: String,
        pub gas: String,
        #[serde(rename = "gasPrice")]
        pub gas_price: Option<String>,
        #[serde(rename = "maxFeePerGas")]
        pub max_fee_per_gas: Option<String>,
        #[serde(rename = "maxPriorityFeePerGas")]
        pub max_priority_fee_per_gas: Option<String>,
        pub input: String,
        pub nonce: String,
        pub v: String,
        pub r: String,
        pub s: String,
        #[serde(rename = "transactionIndex")]
        pub transaction_index: String,
        pub hash: H256,
    }
}

/// Create a simple ETH transfer transaction
fn create_transfer_tx(from: Address, to: Address, value: U256) -> ethrex_common::types::Transaction {
    let tx = ethrex_common::types::LegacyTransaction {
        nonce: 0,
        gas_price: 10_000_000_000u64.into(),
        gas: 21000,
        to: ethrex_common::types::TxKind::Call(to),
        value,
        data: Bytes::new(),
        v: U256::zero(),
        r: U256::zero(),
        s: U256::zero(),
        inner_hash: once_cell::sync::OnceCell::new(),
        sender_cache: once_cell::sync::OnceCell::new(),
    };
    ethrex_common::types::Transaction::LegacyTransaction(tx)
}

use rpc::RpcClient;

fn main() {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <block_number> [rpc_url]", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} 20000000", args[0]);
        eprintln!("  {} 20000000 https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY", args[0]);
        eprintln!();
        eprintln!("Default RPC URL: https://eth.llamarpc.com (public, rate-limited)");
        std::process::exit(1);
    }

    let block_number: u64 = args[1].parse().expect("Invalid block number");
    let rpc_url = args.get(2).map(|s| s.as_str()).unwrap_or("https://eth.llamarpc.com");

    println!("=== Real Ethereum Block Proving Pipeline ===\n");
    println!("Block: {}", block_number);
    println!("RPC: {}\n", rpc_url);

    // ============================================================
    // STEP 1: Connect to RPC and fetch block data
    // ============================================================
    println!("STEP 1: Fetching block {} from RPC...\n", block_number);

    let client = RpcClient::new(rpc_url).expect("Failed to create RPC client");

    let block = match client.get_block_by_number(block_number) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to fetch block: {}", e);
            eprintln!("Hint: The public RPC might be rate-limited. Try using Infura/Alchemy:");
            eprintln!("  {} {} https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY", args[0], block_number);
            std::process::exit(1);
        }
    };

    println!("  Block hash: {:?}", block.hash);
    println!("  Parent hash: {:?}", block.parent_hash);
    println!("  Transactions: {}", block.transactions.len());

    // ============================================================
    // STEP 2: Select a transaction to prove
    // ============================================================
    println!("\nSTEP 2: Selecting transaction to prove...\n");

    // For simplicity, we pick the first simple transfer (ETH only, no data)
    // that has a non-zero value and is a call (not contract creation)
    let tx = block.transactions.iter()
        .find(|tx| {
            tx.to.is_some() &&
            tx.input.is_empty() &&
            !parse_u256_hex(&tx.value).unwrap_or(U256::zero()).is_zero()
        })
        .or_else(|| {
            // Fallback: first transaction with a recipient
            block.transactions.iter().find(|tx| tx.to.is_some())
        })
        .or_else(|| block.transactions.first());

    let tx = match tx {
        Some(tx) => tx,
        None => {
            eprintln!("No suitable transaction found in block");
            std::process::exit(1);
        }
    };

    let to_address = tx.to.unwrap_or(Address::zero());
    let from_address = tx.from;

    // Get the transaction's nonce - needed for correct account setup
    let tx_nonce = parse_u64_hex(&tx.nonce).unwrap_or(0);

    println!("  From: {:?}", from_address);
    println!("  To: {:?}", to_address);
    println!("  Value: {}", parse_u256_hex(&tx.value).unwrap_or(U256::zero()));
    println!("  Input: {} bytes", tx.input.len());
    println!("  Transaction nonce: {}", tx_nonce);

    // ============================================================
    // STEP 3: Fetch account states
    // ============================================================
    println!("\nSTEP 3: Fetching account states...\n");

    // Fetch sender state
    let sender_balance = client.get_balance(from_address, block_number).unwrap_or(U256::zero());
    let sender_nonce = client.get_nonce(from_address, block_number).unwrap_or(0);

    // If both are 0, the RPC likely doesn't support historical state queries
    let use_fallback = sender_balance.is_zero() && sender_nonce == 0;

    if use_fallback {
        println!("  WARNING: RPC returned balance=0, nonce=0 - historical state not supported");
        println!("  Using synthetic transaction with known test accounts instead");
        println!();

        // Use known test accounts with 10 ETH each
        let test_sender = Address::from_str("0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266").unwrap();
        let test_recipient = Address::from_str("0x70997970C51812dc3A010C7d01b50e0d17dc79C8").unwrap();

        println!("  Using test transaction:");
        println!("    From: {:?}", test_sender);
        println!("    To: {:?}", test_recipient);
        println!("    Value: 1 ETH");
        println!();

        // Set up test accounts with proper state
        let mut db = RealisticDatabase::new();
        let test_balance = U256::from(10u64.pow(18));
        db.setup_account(test_sender, test_balance, 0, H256::zero());
        db.setup_account(test_recipient, U256::zero(), 0, H256::zero());

        // Set block hashes
        for i in 0..256u64 {
            let hash = H256::from_uint(&U256::from(i));
            db.set_block_hash(i, hash);
        }

        // Build test transaction (simple ETH transfer)
        let tx = create_transfer_tx(test_sender, test_recipient, U256::one() * U256::from(10u64.pow(17))); // 0.1 ETH

        // Build block header for environment
        let block_header = BlockHeader {
            difficulty: U256::from(2u64.pow(31)),
            number: block_number,
            gas_limit: 30_000_000,
            timestamp: 1717281407u64,
            prev_randao: H256::zero(),
            base_fee_per_gas: Some(4936957716u64),
            ..Default::default()
        };

        // Execute the test transaction
        println!("STEP 5: Setting up EVM environment...\n");
        println!("  (Using synthetic test transaction)");
        println!("  Block: #{}", block_number);
        println!("  Gas limit: {}", block_header.gas_limit);

        let db = Arc::new(db);
        let mut gen_db = GeneralizedDatabase::new(db);

        let chain_config = gen_db.store.get_chain_config().unwrap();
        let config = EVMConfig::new_from_chain_config(&chain_config, &block_header);

        let env = ethrex_levm::environment::Environment {
            origin: test_sender,
            gas_limit: tx.gas_limit(),
            config,
            block_number: block_header.number,
            coinbase: Address::zero(),
            timestamp: block_header.timestamp,
            prev_randao: Some(block_header.prev_randao),
            slot_number: U256::zero(),
            chain_id: U256::from(1),
            base_fee_per_gas: U256::from(block_header.base_fee_per_gas.unwrap_or(4936957716u64)),
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

        println!();
        println!("STEP 6: Executing transaction on ethrex VM...\n");
        println!("  Execution completed in {:?}", exec_time);
        println!("  Success: {}", success);
        println!("  Gas used: {}", final_gas);
        println!("  Trace: {} steps, {} bytes", steps_len, trace_data.len());

        if steps_len == 0 {
            eprintln!("  No trace steps generated");
            std::process::exit(1);
        }

        // Compute bytecode hash (empty for simple transfer)
        let bytecode_hash_arr = [0u8; 32];

        println!();
        println!("STEP 7: Building Jolt prover...\n");

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

        println!();
        println!("STEP 8: Generating Jolt proof...\n");

        let proof_start = Instant::now();
        let (output, proof, io_device) = prover(
            0u64,                      // entry_pc
            bytecode_hash_arr,          // bytecode_hash
            steps_len,                 // steps_len
            final_gas,                 // final_gas
            success,                   // success
            &trace_data,               // trace_data
        );
        let proof_time = proof_start.elapsed();

        println!("  Proof generated in {:.3}s", proof_time.as_secs_f64());

        println!();
        println!("STEP 9: Verifying proof...\n");

        let is_valid = verifier(
            0u64,
            bytecode_hash_arr,
            steps_len,
            final_gas,
            success,
            &trace_data,
            output,
            io_device.panic,
            proof,
        );

        println!("  Verification result: {}", is_valid);

        println!("\n{}", "=".repeat(60));
        println!("SUMMARY");
        println!("{}", "=".repeat(60));
        println!();
        println!("Block: #{} (using synthetic test transaction)", block_number);
        println!("Transaction: ETH transfer");
        println!("  From: {:?}", test_sender);
        println!("  To: {:?}", test_recipient);
        println!("  Value: 0.1 ETH");
        println!("  Success: {}", success);
        println!("  Gas used: {}", final_gas);
        println!("  Trace: {} steps", steps_len);
        println!();
        println!("Timings:");
        println!("  VM execution: {:?}", exec_time);
        println!("  Prover setup: {:.3}s", setup_time.as_secs_f64());
        println!("  Proof generation: {:.3}s", proof_time.as_secs_f64());
        println!();
        println!("Result: {}", if is_valid { "PROOF VERIFIED" } else { "VERIFICATION FAILED" });

        std::process::exit(0);
    }

    println!("  Sender balance: {}", sender_balance);
    println!("  Sender nonce: {}", sender_nonce);

    // Fetch recipient state (if it's a contract)
    let recipient_code = if to_address != Address::zero() {
        match client.get_code(to_address, block_number) {
            Ok(code) => {
                println!("  Recipient has code: {} bytes", code.len());
                code
            }
            Err(_) => Bytes::new(),
        }
    } else {
        Bytes::new()
    };

    // ============================================================
    // STEP 4: Build database with fetched state
    // ============================================================
    println!("\nSTEP 4: Building EVM state...\n");

    let db = RealisticDatabase::new();

    // Set up sender - use tx_nonce (the nonce at time of execution), not sender_nonce (current nonce)
    db.setup_account(from_address, sender_balance, tx_nonce, H256::zero());

    // Set up recipient contract if exists
    let bytecode_hash = if !recipient_code.is_empty() {
        let hash = db.setup_contract(to_address, recipient_code.clone());
        println!("  Contract code hash: {:?}", hash);
        hash
    } else {
        H256::zero()
    };

    // Set block hashes for BLOCKHASH opcode
    // For a real block, we'd need historical block hashes - for now use simple hashes
    for i in 0..256u64 {
        let hash = H256::from_uint(&U256::from(i));
        db.set_block_hash(i, hash);
    }

    // ============================================================
    // STEP 5: Build EVM environment and transaction
    // ============================================================
    println!("\nSTEP 5: Setting up EVM environment...\n");

    // Parse block header values
    let difficulty = parse_u256_hex(&block.difficulty).unwrap_or(U256::from(2u64.pow(31)));
    let gas_limit = parse_u64_hex(&block.gas_limit).unwrap_or(30_000_000);
    let timestamp = parse_u64_hex(&block.timestamp).unwrap_or(0);
    let block_number_u64 = parse_u64_hex(&block.number).unwrap_or(0);
    let base_fee = block.base_fee_per_gas
        .as_ref()
        .and_then(|f| parse_u64_hex(f))
        .unwrap_or(10_000_000_000u64);

    let block_header = BlockHeader {
        difficulty,
        number: block_number_u64,
        gas_limit,
        timestamp,
        prev_randao: block.prev_randao.unwrap_or_default(),
        base_fee_per_gas: Some(base_fee),
        ..Default::default()
    };

    // Build transaction
    let input = hex::decode(tx.input.trim_start_matches("0x"))
        .map(|v| Bytes::from(v))
        .unwrap_or_default();

    let value = parse_u256_hex(&tx.value).unwrap_or(U256::zero());
    let gas = parse_u64_hex(&tx.gas).unwrap_or(21000);

    // Create transaction object
    let chain_id = 1u64;
    let gas_price = tx.gas_price.as_ref()
        .and_then(|f| parse_u64_hex(f))
        .unwrap_or(0);
    let max_fee = tx.max_fee_per_gas.as_ref()
        .and_then(|f| parse_u64_hex(f))
        .unwrap_or(gas_price * 2);
    let max_priority = tx.max_priority_fee_per_gas.as_ref()
        .and_then(|f| parse_u64_hex(f))
        .unwrap_or(0);

    let tx_obj = if let (Some(_), Some(_)) = (tx.max_fee_per_gas.as_ref(), tx.max_priority_fee_per_gas.as_ref()) {
        // EIP-1559 transaction
        let tx = ethrex_common::types::EIP1559Transaction {
            chain_id,
            nonce: tx_nonce,
            max_priority_fee_per_gas: max_priority,
            max_fee_per_gas: max_fee,
            gas_limit: gas,
            to: ethrex_common::types::TxKind::Call(to_address),
            value,
            data: input.clone(),
            access_list: vec![],
            signature_y_parity: false,
            signature_r: U256::zero(),
            signature_s: U256::zero(),
            inner_hash: once_cell::sync::OnceCell::new(),
            sender_cache: once_cell::sync::OnceCell::new(),
            cached_canonical: once_cell::sync::OnceCell::new(),
        };
        ethrex_common::types::Transaction::EIP1559Transaction(tx)
    } else {
        // Legacy transaction
        let tx = ethrex_common::types::LegacyTransaction {
            nonce: tx_nonce,
            gas_price: U256::from(gas_price),
            gas,
            to: ethrex_common::types::TxKind::Call(to_address),
            value,
            data: input.clone(),
            v: U256::zero(),
            r: U256::zero(),
            s: U256::zero(),
            inner_hash: once_cell::sync::OnceCell::new(),
            sender_cache: once_cell::sync::OnceCell::new(),
        };
        ethrex_common::types::Transaction::LegacyTransaction(tx)
    };

    println!("  Block: #{} ({})", block_header.number, block_header.timestamp);
    println!("  Gas limit: {}", block_header.gas_limit);
    println!("  Base fee: {}", base_fee);

    // ============================================================
    // STEP 6: Execute transaction on ethrex VM
    // ============================================================
    println!("\nSTEP 6: Executing transaction on ethrex VM...\n");

    let db = Arc::new(db);
    let mut gen_db = GeneralizedDatabase::new(db);

    let chain_config = gen_db.store.get_chain_config().unwrap();
    let config = EVMConfig::new_from_chain_config(&chain_config, &block_header);

    let env = ethrex_levm::environment::Environment {
        origin: from_address,
        gas_limit: tx_obj.gas_limit(),
        config,
        block_number: block_header.number,
        coinbase: Address::zero(),
        timestamp: block_header.timestamp,
        prev_randao: Some(block_header.prev_randao),
        slot_number: U256::zero(),
        chain_id: U256::from(chain_id),
        base_fee_per_gas: U256::from(base_fee),
        base_blob_fee_per_gas: U256::zero(),
        gas_price: tx_obj.gas_price(),
        block_excess_blob_gas: None,
        block_blob_gas_used: None,
        tx_blob_hashes: vec![],
        tx_max_priority_fee_per_gas: tx_obj.max_priority_fee().map(U256::from),
        tx_max_fee_per_gas: tx_obj.max_fee_per_gas().map(U256::from),
        tx_max_fee_per_blob_gas: tx_obj.max_fee_per_blob_gas(),
        tx_nonce: tx_obj.nonce(),
        block_gas_limit: block_header.gas_limit,
        difficulty: block_header.difficulty,
        is_privileged: false,
        fee_token: None,
        disable_balance_check: false,
    };

    let mut vm = VM::new(
        env,
        &mut gen_db,
        &tx_obj,
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

    // If no steps, exit early
    if steps_len == 0 {
        println!("\n  No trace steps - transaction may have failed early.");
        std::process::exit(1);
    }

    // ============================================================
    // STEP 7: Build Jolt prover
    // ============================================================
    println!("\nSTEP 7: Building Jolt prover...\n");

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
    // STEP 8: Generate proof
    // ============================================================
    println!("\nSTEP 8: Generating Jolt proof...\n");

    // Compute bytecode hash
    let bytecode_hash_arr = keccak256(&recipient_code);
    let entry_pc = 0u64;

    let proof_start = Instant::now();
    let (output, proof, io_device) = prover(
        entry_pc,
        bytecode_hash_arr,
        steps_len,
        final_gas,
        success,
        &trace_data,
    );
    let proof_time = proof_start.elapsed();

    println!("  Proof generated in {:.3}s", proof_time.as_secs_f64());

    // ============================================================
    // STEP 9: Verify proof
    // ============================================================
    println!("\nSTEP 9: Verifying proof...\n");

    let is_valid = verifier(
        entry_pc,
        bytecode_hash_arr,
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
    println!("Block: #{}", block_number);
    println!("Transaction: {:?}", tx.hash);
    println!("  From: {:?}", from_address);
    println!("  To: {:?}", to_address);
    println!("  Value: {}", value);
    println!("  Input: {} bytes", input.len());
    println!("  Success: {}", success);
    println!("  Gas used: {}", final_gas);
    println!("  Trace: {} steps", steps_len);
    println!();
    println!("Timings:");
    println!("  VM execution: {:?}", exec_time);
    println!("  Prover setup: {:.3}s", setup_time.as_secs_f64());
    println!("  Proof generation: {:.3}s", proof_time.as_secs_f64());
    println!();
    println!("Result: {}", if is_valid { "PROOF VERIFIED" } else { "VERIFICATION FAILED" });
}

// Helper functions
fn parse_u64_hex(hex: &str) -> Option<u64> {
    let hex = hex.trim().trim_start_matches("0x");
    if hex.is_empty() {
        return Some(0);
    }
    u64::from_str_radix(hex, 16).ok()
}

fn parse_u256_hex(hex: &str) -> Option<U256> {
    let hex = hex.trim().trim_start_matches("0x");
    if hex.is_empty() {
        return Some(U256::zero());
    }
    U256::from_str_radix(hex, 16).ok()
}
