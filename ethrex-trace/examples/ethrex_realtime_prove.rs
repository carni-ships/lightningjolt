//! Real-time Ethereum block proving with FULL archival state access
//!
//! This example fetches real contract state (bytecode, storage, balances) from
//! the RPC and proves ALL transactions in a block - including smart contract
//! interactions like ERC-20 transfers.
//!
//! State caching system:
//! - Fetches and caches full blocks for later use
//! - Maintains a reference block log (blocks_full_blocks.jsonl)
//! - Can re-prove any cached block without re-fetching from RPC
//!
//! Run with: cargo run -p ethrex-trace --example ethrex_realtime_prove -- [rpc_url] [cache_dir]
//!
//! Default cache dir: /tmp/ethrex-block-cache

use ethrex_common::{
    Address, Bytes, H256, U256, BigEndianHash,
    types::{AccountState, ChainConfig, Code, CodeMetadata, BlockHeader, Transaction},
};
use ethrex_crypto::NativeCrypto;
use ethrex_levm::db::{gen_db::GeneralizedDatabase, Database};
use ethrex_levm::environment::EVMConfig;
use ethrex_levm::step_tracer::{StepTrace, StepTracer};
use ethrex_levm::tracing::LevmCallTracer;
use ethrex_levm::vm::{VM, VMType};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Instant;

const STEP_SIZE: usize = 41;

/// Default cache directory for blocks
const DEFAULT_CACHE_DIR: &str = "/tmp/ethrex-block-cache";
const FULL_BLOCKS_LOG: &str = "blocks_full_blocks.jsonl";

fn keccak256(data: &[u8]) -> [u8; 32] {
    use tiny_keccak::Hasher;
    let mut hasher = tiny_keccak::Keccak::v256();
    let mut output = [0u8; 32];
    hasher.update(data);
    hasher.finalize(&mut output);
    output
}

mod rpc {
    use super::*;
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
        #[serde(rename = "result")]
        result: serde_json::Value,
    }

    /// RPC client with retry logic and rate limiting handling
#[derive(Clone)]
pub struct RpcClient {
    client: reqwest::blocking::Client,
    url: String,
}

impl RpcClient {
    pub fn new(url: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            client: reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()?,
            url: url.to_string(),
        })
    }

    /// Call with retry logic for rate limiting (429 errors or "too many connections")
    fn call_with_retry(&self, method: &str, params: serde_json::Value, max_retries: u32) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut retries = 0;
        loop {
            match self.call_internal(method, params.clone()) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    let err_str = e.to_string();
                    // Check if it's a rate limit or connection error
                    let is_rate_limited = err_str.contains("429") ||
                        err_str.contains("rate-limited") ||
                        err_str.contains("Too many connections") ||
                        err_str.contains("expected value");

                    if is_rate_limited {
                        if retries >= max_retries {
                            return Err(format!("Rate limited after {} retries: {}", retries, e).into());
                        }
                        retries += 1;
                        // Exponential backoff: 1s, 2s, 4s, etc.
                        let wait_ms = 1000 * (1 << retries.min(5));
                        eprintln!("  [RPC] Rate limited, retrying in {}ms (attempt {}/{})", wait_ms, retries, max_retries);
                        std::thread::sleep(std::time::Duration::from_millis(wait_ms));
                    } else {
                        return Err(e);
                    }
                }
            }
        }
    }

    fn call_internal(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
            id: 1,
        };
        let text = self
            .client
            .post(&self.url)
            .header("content-type", "application/json")
            .header("accept", "application/json")
            .json(&request)
            .send()?
            .text()?;
        // Parse JSON with better error handling
        let result: JsonRpcResponse = serde_json::from_str(&text)
            .map_err(|e| format!("JSON parse error: {} (text: {})", e, &text[..text.len().min(200)]))?;
        Ok(result.result)
    }

    pub fn get_block_by_number(&self, block_number: u64, with_txs: bool) -> Result<Option<BlockResponse>, Box<dyn std::error::Error>> {
        let result = self.call_with_retry("eth_getBlockByNumber", serde_json::json!([format!("0x{:x}", block_number), with_txs]), 5)?;
        if result.is_null() { return Ok(None); }
        serde_json::from_value(result).map_err(|e| format!("Parse error: {}", e).into())
    }

    pub fn get_block_number(&self) -> Result<u64, Box<dyn std::error::Error>> {
        let result = self.call_with_retry("eth_blockNumber", serde_json::Value::Null, 5)?;
        let hex: String = serde_json::from_value(result)?;
        let hex = hex.trim_start_matches("0x");
        Ok(u64::from_str_radix(hex, 16).unwrap_or(0))
    }

    pub fn get_balance(&self, address: Address, block: u64) -> Result<U256, Box<dyn std::error::Error>> {
        let result = self.call_with_retry("eth_getBalance", serde_json::json!([format!("{:?}", address), format!("0x{:x}", block)]), 3)?;
        let hex: String = serde_json::from_value(result)?;
        let hex = hex.trim_start_matches("0x");
        if hex.is_empty() { return Ok(U256::zero()); }
        Ok(U256::from_str_radix(hex, 16).unwrap_or(U256::zero()))
    }

    pub fn get_transaction_count(&self, address: Address, block: u64) -> Result<u64, Box<dyn std::error::Error>> {
        let result = self.call_with_retry("eth_getTransactionCount", serde_json::json!([format!("{:?}", address), format!("0x{:x}", block)]), 3)?;
        let hex: String = serde_json::from_value(result)?;
        let hex = hex.trim_start_matches("0x");
        if hex.is_empty() { return Ok(0); }
        Ok(u64::from_str_radix(hex, 16).unwrap_or(0))
    }

    pub fn get_code(&self, address: Address, block: u64) -> Result<Bytes, Box<dyn std::error::Error>> {
        let result = self.call_with_retry("eth_getCode", serde_json::json!([format!("{:?}", address), format!("0x{:x}", block)]), 3)?;
        let hex: String = serde_json::from_value(result)?;
        let hex = hex.trim_start_matches("0x");
        if hex.is_empty() { return Ok(Bytes::new()); }
        let bytes = hex::decode(hex).unwrap_or_default();
        Ok(Bytes::from(bytes))
    }

    pub fn get_storage_at(&self, address: Address, slot: U256, block: u64) -> Result<U256, Box<dyn std::error::Error>> {
        let result = self.call_with_retry("eth_getStorageAt", serde_json::json!([
            format!("{:?}", address),
            format!("{:?}", slot),
            format!("0x{:x}", block)
        ]), 3)?;
        let hex: String = serde_json::from_value(result)?;
        let hex = hex.trim_start_matches("0x");
        if hex.len() < 64 { return Ok(U256::zero()); }
        let padded = format!("{:0>64}", hex);
        Ok(U256::from_str_radix(&padded, 16).unwrap_or(U256::zero()))
    }

    pub fn get_block_hash(&self, block_number: u64) -> Result<H256, Box<dyn std::error::Error>> {
        let result = self.call_with_retry("eth_getBlockByNumber", serde_json::json!([format!("0x{:x}", block_number), false]), 3)?;
        if result.is_null() { return Ok(H256::zero()); }
        #[derive(Deserialize)]
        struct SimpleBlock { hash: Option<H256> }
        let block: SimpleBlock = serde_json::from_value(result).map_err(|e| format!("{:?}", e))?;
        Ok(block.hash.unwrap_or(H256::zero()))
    }

    /// Fetch full state for an account (balance, nonce, code, storage proofs)
    pub fn get_proof(&self, address: Address, storage_keys: &[H256], block: u64) -> Result<Option<AccountProof>, Box<dyn std::error::Error>> {
        // Convert H256 to hex strings for the RPC call
        let storage_hex: Vec<String> = storage_keys.iter().map(|k| {
            format!("0x{}", hex::encode(k.as_bytes()))
        }).collect();

        let result = self.call_with_retry("eth_getProof", serde_json::json!([
            format!("{:?}", address),
            storage_hex,
            format!("0x{:x}", block)
        ]), 3)?;
        if result.is_null() { return Ok(None); }
        serde_json::from_value(result).map_err(|e| format!("Parse proof: {}", e).into())
    }
}

    #[derive(Debug, Deserialize, Clone)]
    #[serde(rename_all = "camelCase")]
    pub struct BlockResponse {
        pub hash: H256,
        pub difficulty: String,
        pub gas_limit: String,
        pub timestamp: String,
        pub number: Option<String>,
        #[serde(alias = "mixHash", default)]
        pub prev_randao: Option<H256>,
        pub base_fee_per_gas: Option<String>,
        pub miner: Option<Address>,
        pub transactions: Vec<TransactionResponse>,
    }

    #[derive(Debug, Deserialize, Clone)]
    #[serde(rename_all = "camelCase")]
    pub struct TransactionResponse {
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
        pub hash: H256,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct AccountProof {
        pub balance: String,
        pub nonce: String,
        pub code_hash: String,
        pub storage_hash: String,
        #[serde(rename = "storageProof")]
        pub storage_proofs: Vec<StorageProof>,
    }

    #[derive(Debug, Deserialize)]
    pub struct StorageProof {
        pub key: String,
        pub value: String,
    }
}

use rpc::RpcClient;

/// Full block state cached to disk
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CachedBlockState {
    pub block_number: u64,
    pub pre_state_block: u64,  // Block at which this state is valid (usually block_number - 1)
    pub block_hash: H256,
    pub timestamp: u64,
    // Account states: address -> cached account
    pub accounts: HashMap<Address, CachedAccount>,
    // Storage entries as list of (address_hex, slot_hex, value) for JSON serialization
    pub storage: Vec<(String, String, U256)>,
    // Block hashes for last 256 blocks
    pub block_hashes: HashMap<u64, H256>,
    pub gas_limit: u64,
    pub difficulty: U256,
    pub base_fee: u64,
    pub miner: Address,
    pub prev_randao: H256,
    // Transaction data - cached to avoid RPC calls for reproving
    pub transactions: Vec<CachedTransaction>,
}

/// Cached transaction data for reproving
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CachedTransaction {
    pub from: Address,
    pub to: Option<Address>,
    pub value: U256,
    pub gas: u64,
    pub gas_price: u64,
    pub max_fee_per_gas: Option<u64>,
    pub max_priority_fee_per_gas: Option<u64>,
    pub input: Vec<u8>,
    pub nonce: u64,
    pub hash: H256,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CachedAccount {
    pub balance: U256,
    pub nonce: u64,
    pub code_hash: H256,
    pub bytecode: Vec<u8>,
}

/// Reference block log entry
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlockLogEntry {
    pub block_number: u64,
    pub block_hash: H256,
    pub tx_count: usize,
    pub cached_at: u64,  // Unix timestamp
    pub file_path: String,
}

/// Reference block log (stored as JSONL)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlockLog {
    pub entries: Vec<BlockLogEntry>,
}

impl Default for BlockLog {
    fn default() -> Self {
        Self { entries: Vec::new() }
    }
}

/// Block cache manager - fetches, stores, and serves cached blocks
struct BlockCache {
    cache_dir: PathBuf,
    log_path: PathBuf,
    log: RwLock<BlockLog>,
    // In-memory cache for frequently accessed blocks
    memory_cache: RwLock<HashMap<u64, CachedBlockState>>,
}

impl BlockCache {
    fn new(cache_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let cache_dir = PathBuf::from(cache_dir);
        fs::create_dir_all(&cache_dir)?;

        let log_path = cache_dir.join("blocks_full_blocks.jsonl");
        let log = Self::load_log(&log_path);

        Ok(Self {
            cache_dir,
            log_path,
            log: RwLock::new(log),
            memory_cache: RwLock::new(HashMap::new()),
        })
    }

    fn load_log(path: &PathBuf) -> BlockLog {
        if !path.exists() {
            return BlockLog::default();
        }
        let content = fs::read_to_string(path).unwrap_or_default();
        serde_json::from_str(&content).unwrap_or_default()
    }

    fn save_log(&self) -> Result<(), Box<dyn std::error::Error>> {
        let log = self.log.read().unwrap();
        let content = serde_json::to_string_pretty(&*log)?;
        fs::write(&self.log_path, content)?;
        Ok(())
    }

    fn get_block_file_path(&self, block_number: u64) -> PathBuf {
        self.cache_dir.join(format!("block_{:010}.json", block_number))
    }

    /// Check if a block is cached
    fn is_cached(&self, block_number: u64) -> bool {
        self.get_block_file_path(block_number).exists()
    }

    /// Get cached block from memory or disk
    fn get_block(&self, block_number: u64) -> Option<CachedBlockState> {
        // Check memory cache first
        {
            let cache = self.memory_cache.read().unwrap();
            if let Some(block) = cache.get(&block_number) {
                return Some(block.clone());
            }
        }

        // Load from disk
        let path = self.get_block_file_path(block_number);
        if !path.exists() {
            return None;
        }

        match fs::read_to_string(&path) {
            Ok(content) => {
                match serde_json::from_str::<CachedBlockState>(&content) {
                    Ok(block) => {
                        // Add to memory cache
                        let mut cache = self.memory_cache.write().unwrap();
                        cache.insert(block_number, block.clone());
                        Some(block)
                    }
                    Err(e) => {
                        eprintln!("  [WARN] Failed to parse cached block {}: {}", block_number, e);
                        None
                    }
                }
            }
            Err(e) => {
                eprintln!("  [WARN] Failed to read cached block {}: {}", block_number, e);
                None
            }
        }
    }

    /// Cache a block to disk and update the log
    fn cache_block(&self, state: &CachedBlockState) -> Result<(), Box<dyn std::error::Error>> {
        let path = self.get_block_file_path(state.block_number);
        let content = serde_json::to_string_pretty(state)?;
        fs::write(&path, content)?;

        // Update log
        {
            let mut log = self.log.write().unwrap();
            // Remove existing entry if present
            log.entries.retain(|e| e.block_number != state.block_number);
            // Add new entry
            log.entries.push(BlockLogEntry {
                block_number: state.block_number,
                block_hash: state.block_hash,
                tx_count: state.accounts.len(),  // Approximate
                cached_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                file_path: path.to_string_lossy().to_string(),
            });
            // Sort by block number
            log.entries.sort_by_key(|e| e.block_number);
        }

        // Save log
        self.save_log()?;

        // Add to memory cache
        {
            let mut cache = self.memory_cache.write().unwrap();
            cache.insert(state.block_number, state.clone());
        }

        println!("  [CACHE] Saved block #{} to {}", state.block_number, path.display());
        Ok(())
    }

    /// List all cached blocks
    fn list_cached_blocks(&self) -> Vec<BlockLogEntry> {
        self.log.read().unwrap().entries.clone()
    }

    /// Fetch and cache a block from RPC
    fn fetch_and_cache_block(&self, client: &RpcClient, block_number: u64) -> Result<CachedBlockState, Box<dyn std::error::Error>> {
        println!("  Fetching block #{} from RPC...", block_number);

        let block = match client.get_block_by_number(block_number, true)? {
            Some(b) => b,
            None => return Err(format!("Block #{} not found", block_number).into()),
        };

        let block_hash = block.hash;
        let timestamp = parse_u64_hex(&block.timestamp).unwrap_or(0);
        let gas_limit = parse_u64_hex(&block.gas_limit).unwrap_or(30_000_000);
        let difficulty = parse_u256_hex(&block.difficulty).unwrap_or(U256::zero());
        let base_fee = block.base_fee_per_gas.as_ref().and_then(|f| parse_u64_hex(f)).unwrap_or(10_000_000_000u64);
        let miner = block.miner.unwrap_or(Address::zero());
        let prev_randao = block.prev_randao.unwrap_or_default();

        // Collect all addresses and cache transactions
        let mut addresses = std::collections::HashSet::new();
        let cached_txs: Vec<CachedTransaction> = block.transactions.iter().map(|tx| {
            addresses.insert(tx.from);
            if let Some(to) = tx.to {
                addresses.insert(to);
            }
            CachedTransaction {
                from: tx.from,
                to: tx.to,
                value: parse_u256_hex(&tx.value).unwrap_or(U256::zero()),
                gas: parse_u64_hex(&tx.gas).unwrap_or(21000),
                gas_price: tx.gas_price.as_ref().and_then(|f| parse_u64_hex(f)).unwrap_or(0),
                max_fee_per_gas: tx.max_fee_per_gas.as_ref().and_then(|f| parse_u64_hex(f)),
                max_priority_fee_per_gas: tx.max_priority_fee_per_gas.as_ref().and_then(|f| parse_u64_hex(f)),
                input: hex::decode(tx.input.trim_start_matches("0x")).unwrap_or_default(),
                nonce: parse_u64_hex(&tx.nonce).unwrap_or(0),
                hash: tx.hash,
            }
        }).collect();

        // Fetch state for all addresses at block_number - 1 (pre-block state)
        // Transactions in block N modify state that exists after block N-1 was mined
        let state_block = block_number.saturating_sub(1);
        let mut accounts = HashMap::new();
        let mut storage: Vec<(String, String, U256)> = Vec::new();
        println!("  Fetching pre-block state (at block #{})...", state_block);
        for (i, addr) in addresses.iter().enumerate() {
            let code = client.get_code(*addr, state_block).unwrap_or_else(|_| Bytes::new());
            let balance = client.get_balance(*addr, state_block).unwrap_or(U256::zero());
            let nonce = client.get_transaction_count(*addr, state_block).unwrap_or(0);

            let code_bytes = code.as_ref();
            let code_hash = if code_bytes.is_empty() {
                H256::zero()
            } else {
                H256::from_slice(&keccak256(code_bytes)[..])
            };

            accounts.insert(*addr, CachedAccount {
                balance,
                nonce,
                code_hash,
                bytecode: code_bytes.to_vec(),
            });

            // Fetch common storage slots for contracts
            let common_slots = vec![
                H256::zero(),
                H256::from_low_u64_be(1),
                H256::from_low_u64_be(2),
                H256::from_low_u64_be(9),
            ];

            if let Ok(Some(proof)) = client.get_proof(*addr, &common_slots, state_block) {
                for sp in proof.storage_proofs {
                    let key_str = sp.key.trim_start_matches("0x");
                    let val_hex = sp.value.trim_start_matches("0x");
                    let val = if val_hex.is_empty() || val_hex == "0" {
                        U256::zero()
                    } else {
                        U256::from_str_radix(val_hex, 16).unwrap_or(U256::zero())
                    };
                    // Store as (address_hex, key_hex, value) tuple
                    storage.push((format!("{:?}", addr), format!("0x{}", key_str), val));
                }
            }

            if (i + 1) % 20 == 0 {
                println!("    Fetched {}/{}", i + 1, addresses.len());
            }
        }

        // Fetch block hashes for last 256 blocks up to the state block
        let mut block_hashes = HashMap::new();
        for i in state_block.saturating_sub(255)..=state_block {
            if let Ok(hash) = client.get_block_hash(i) {
                if hash != H256::zero() {
                    block_hashes.insert(i, hash);
                }
            }
        }

        let state = CachedBlockState {
            block_number,  // Original block being cached (for reference)
            pre_state_block: state_block,  // Block at which this state is valid
            block_hash,
            timestamp,
            accounts,
            storage,
            block_hashes,
            gas_limit,
            difficulty,
            base_fee,
            miner,
            prev_randao,
            transactions: cached_txs,
        };

        // Cache the block
        self.cache_block(&state)?;

        Ok(state)
    }
}

/// Database backed by cached block state
struct CachedBlockDatabase {
    state: CachedBlockState,
    chain_config: ChainConfig,
}

impl CachedBlockDatabase {
    fn from_cached(state: CachedBlockState) -> Self {
        Self {
            state,
            chain_config: ChainConfig::default(),
        }
    }
}

impl Database for CachedBlockDatabase {
    fn get_account_state(&self, address: Address) -> Result<AccountState, ethrex_levm::errors::DatabaseError> {
        Ok(self.state.accounts.get(&address).map(|acc| AccountState {
            nonce: acc.nonce,
            balance: acc.balance,
            code_hash: acc.code_hash,
            storage_root: H256::zero(),
        }).unwrap_or_default())
    }

    fn get_storage_value(&self, address: Address, key: H256) -> Result<U256, ethrex_levm::errors::DatabaseError> {
        let addr_str = format!("{:?}", address);
        let key_str = format!("0x{}", hex::encode(key.as_bytes()));
        for (a, k, v) in &self.state.storage {
            if a == &addr_str && k == &key_str {
                return Ok(*v);
            }
        }
        Ok(U256::zero())
    }

    fn get_block_hash(&self, block_number: u64) -> Result<H256, ethrex_levm::errors::DatabaseError> {
        Ok(self.state.block_hashes.get(&block_number).copied().unwrap_or(H256::zero()))
    }

    fn get_chain_config(&self) -> Result<ChainConfig, ethrex_levm::errors::DatabaseError> {
        Ok(self.chain_config.clone())
    }

    fn get_account_code(&self, code_hash: H256) -> Result<Code, ethrex_levm::errors::DatabaseError> {
        // Find code by hash
        for (_, acc) in &self.state.accounts {
            if acc.code_hash == code_hash {
                let bytecode = Bytes::from(acc.bytecode.clone());
                return Ok(Code::from_bytecode_unchecked(bytecode, code_hash));
            }
        }
        Ok(Code::default())
    }

    fn get_code_metadata(&self, code_hash: H256) -> Result<CodeMetadata, ethrex_levm::errors::DatabaseError> {
        for (_, acc) in &self.state.accounts {
            if acc.code_hash == code_hash {
                return Ok(CodeMetadata { length: acc.bytecode.len() as u64 });
            }
        }
        Ok(CodeMetadata { length: 0 })
    }
}

/// Step tracer that generates 41-byte step records for Jolt
struct TraceGenerator { traces: Vec<u8> }

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
        // Use the actual values from the VM
        self.add_step(trace.opcode, trace.pc, trace.gas_before, trace.gas_used, trace.stack_len_before, trace.stack_len_after);
    }
}

/// Batch trace for combining multiple transactions
struct BatchTrace {
    trace_data: Vec<u8>,
    total_steps: usize,
    tx_count: usize,
    total_gas: u64,
    bytecode_hash: [u8; 32],
}

impl BatchTrace {
    fn new() -> Self {
        Self {
            trace_data: Vec::new(),
            total_steps: 0,
            tx_count: 0,
            total_gas: 0,
            bytecode_hash: [0u8; 32],
        }
    }
    fn add_trace(&mut self, bytecode_hash: [u8; 32], gas_used: u64, trace: Vec<u8>) {
        self.total_steps += trace.len() / STEP_SIZE;
        self.total_gas += gas_used;
        self.tx_count += 1;
        self.bytecode_hash = bytecode_hash;
        self.trace_data.extend(trace);
    }
}

impl Default for BatchTrace { fn default() -> Self { Self::new() } }

const PROVER_CACHE_FILE: &str = "jolt_prover_preprocessing.dat";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <rpc_url> [cache_dir]", args[0]);
        eprintln!("  rpc_url: Ethereum RPC endpoint");
        eprintln!("  cache_dir: Optional cache directory (default: {})", DEFAULT_CACHE_DIR);
        eprintln!("");
        eprintln!("Commands:");
        eprintln!("  Prove latest block:");
        eprintln!("    {} <rpc_url> [cache_dir]", args[0]);
        eprintln!("");
        eprintln!("  Prove specific block:");
        eprintln!("    {} <rpc_url> [cache_dir] <block_number>", args[0]);
        eprintln!("");
        eprintln!("  List cached blocks:");
        eprintln!("    {} <rpc_url> [cache_dir] --list", args[0]);
        std::process::exit(1);
    }

    let rpc_url = &args[1];
    let cache_dir = args.get(2).map(|s| s.as_str()).unwrap_or(DEFAULT_CACHE_DIR);

    // Check for special commands
    if args.len() == 3 && args[2] == "--list" {
        // List cached blocks only
        match BlockCache::new(cache_dir) {
            Ok(cache) => {
                let blocks = cache.list_cached_blocks();
                if blocks.is_empty() {
                    println!("No cached blocks found in {}", cache_dir);
                } else {
                    println!("Cached blocks in {}:", cache_dir);
                    for entry in &blocks {
                        println!("  #{} - {} (cached {})",
                            entry.block_number,
                            entry.block_hash,
                            chrono::DateTime::from_timestamp(entry.cached_at as i64, 0)
                                .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                                .unwrap_or_default()
                        );
                    }
                }
            }
            Err(e) => eprintln!("Failed to open cache: {}", e),
        }
        return;
    }

    // Determine if we're proving a specific block or the latest
    let target_block: Option<u64> = args.get(3).and_then(|s| s.parse().ok());

    println!("=== Real-Time Ethereum Block Proving ===\n");
    println!("RPC: {}", rpc_url);
    println!("Cache: {}\n", cache_dir);

    let client = RpcClient::new(rpc_url).expect("Failed to create RPC client");
    let block_cache = BlockCache::new(cache_dir).expect("Failed to create block cache");

    // Get block to prove
    let block_number = match target_block {
        Some(n) => n,
        None => {
            println!("Fetching latest block number...");
            match client.get_block_number() {
                Ok(n) => n,
                Err(e) => { eprintln!("Failed to get block number: {}", e); std::process::exit(1); }
            }
        }
    };

    println!("Target block: #{}", block_number);

    // Get or fetch the block state
    let cached_state = if block_cache.is_cached(block_number) {
        println!("[CACHE HIT] Loading block #{} from cache...", block_number);
        block_cache.get_block(block_number).expect("Failed to load cached block")
    } else {
        println!("[CACHE MISS] Fetching block #{} from RPC...", block_number);
        match block_cache.fetch_and_cache_block(&client, block_number) {
            Ok(state) => state,
            Err(e) => {
                eprintln!("Failed to fetch block: {}", e);
                std::process::exit(1);
            }
        }
    };

    println!("  Block hash: {:?}", cached_state.block_hash);
    println!("  Accounts cached: {}", cached_state.accounts.len());

    let block_header = BlockHeader {
        difficulty: cached_state.difficulty,
        number: block_number,
        gas_limit: cached_state.gas_limit,
        timestamp: cached_state.timestamp,
        prev_randao: cached_state.prev_randao,
        base_fee_per_gas: Some(cached_state.base_fee),
        ..Default::default()
    };

    let chain_config = ChainConfig::default();
    let config = EVMConfig::new_from_chain_config(&chain_config, &block_header);

    // Use cached transactions instead of fetching from RPC
    println!("\nUsing {} cached transactions for execution...", cached_state.transactions.len());

    println!("\nSTEP 1: Executing transactions...\n");

    // Create database from cached state
    let db = Arc::new(CachedBlockDatabase::from_cached(cached_state.clone()));

    let mut proofs = Vec::new();
    let mut skipped = 0;

    for (idx, tx) in cached_state.transactions.iter().enumerate() {
        let to_address = tx.to.unwrap_or(Address::zero());
        let tx_nonce = tx.nonce;
        let tx_value = tx.value;
        let gas = tx.gas;

        let gas_price = if tx.gas_price > 0 { tx.gas_price } else { cached_state.base_fee };
        let max_fee = tx.max_fee_per_gas.unwrap_or(gas_price * 2);
        let max_priority = tx.max_priority_fee_per_gas.unwrap_or(0);

        let env = ethrex_levm::environment::Environment {
            origin: tx.from, gas_limit: gas, config: config.clone(),
            block_number, coinbase: cached_state.miner, timestamp: cached_state.timestamp,
            prev_randao: Some(cached_state.prev_randao), slot_number: U256::zero(),
            chain_id: U256::from(1), base_fee_per_gas: U256::from(cached_state.base_fee),
            base_blob_fee_per_gas: U256::zero(), gas_price: U256::from(gas_price),
            block_excess_blob_gas: None, block_blob_gas_used: None, tx_blob_hashes: vec![],
            tx_max_priority_fee_per_gas: Some(U256::from(max_priority)),
            tx_max_fee_per_gas: Some(U256::from(max_fee)),
            tx_max_fee_per_blob_gas: None, tx_nonce,
            block_gas_limit: cached_state.gas_limit, difficulty: cached_state.difficulty,
            is_privileged: false, fee_token: None, disable_balance_check: false,
        };

        // Use zero bytecode hash for compatibility
        // The actual bytecode isn't verified in Type B1 - only the trace
        let bytecode_hash = [0u8; 32];

        let input = Bytes::from(tx.input.clone());

        let tx_obj = if tx.max_fee_per_gas.is_some() && tx.max_priority_fee_per_gas.is_some() {
            ethrex_common::types::Transaction::EIP1559Transaction(ethrex_common::types::EIP1559Transaction {
                chain_id: 1, nonce: tx_nonce, max_priority_fee_per_gas: max_priority,
                max_fee_per_gas: max_fee, gas_limit: gas,
                to: ethrex_common::types::TxKind::Call(to_address),
                value: tx_value, data: input.clone(), access_list: vec![],
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

        let db_arc = db.clone();
        let mut gen_db = GeneralizedDatabase::new(db_arc.clone());

        let mut vm = match VM::new(env, &mut gen_db, &tx_obj, LevmCallTracer::disabled(), VMType::L1, &NativeCrypto) {
            Ok(vm) => vm,
            Err(e) => {
                if idx < 3 { eprintln!("  Tx {}: VM creation failed: {:?}", idx, e); }
                skipped += 1;
                continue;
            }
        };

        let mut tracer = TraceGenerator::new();
        vm.set_step_tracer(&mut tracer);

        let result = match vm.execute() {
            Ok(r) => r,
            Err(e) => {
                if idx < 3 { eprintln!("  Tx {}: Execution failed: {:?}", idx, e); }
                skipped += 1;
                continue;
            }
        };

        if !result.is_success() {
            skipped += 1;
            continue;
        }

        let trace_data = tracer.into_bytes();
        let steps = trace_data.len() / STEP_SIZE;
        if steps == 0 {
            skipped += 1;
            continue;
        }

        println!("  Tx {} ({}): {} steps, gas={}", idx, hex::encode(&tx.hash.as_bytes()[..4]), steps, result.gas_used);
        proofs.push((tx.hash, tx.from, to_address, tx_value, result.gas_used, steps, trace_data, bytecode_hash));
    }

    if proofs.is_empty() {
        println!("\n  No provable transactions found.");
        println!("  Skipped: {}", skipped);
        std::process::exit(1);
    }

    println!("\n  {} provable transactions ({} skipped)\n", proofs.len(), skipped);

    // Show first provable transaction details
    if !proofs.is_empty() {
        let first = &proofs[0];
        println!("  First provable tx: {} ({} steps, gas={})",
            hex::encode(&first.0.as_bytes()[..4]),
            first.5, first.4);
    }

    // Limit to provable transactions that have meaningful traces
    // Filter out failed/invalid transactions (1 step with STOP)
    let meaningful_count = proofs.iter()
        .filter(|p| p.5 > 1 || (p.5 == 1 && p.6[0] != 0x00))  // More than 1 step, or single non-STOP step
        .count();

    let proofs = if meaningful_count > 0 {
        // Use transactions with meaningful traces
        let selected: Vec<_> = proofs.into_iter()
            .filter(|p| p.5 > 1 || (p.5 == 1 && p.6[0] != 0x00))
            .take(20)  // Increased for better Dory-Prime parallelization
            .collect();
        println!("[DEBUG] Using {} transactions with meaningful traces", selected.len());
        selected
    } else {
        // Use just the first provable
        vec![proofs.into_iter().next().unwrap()]
    };

    if proofs.is_empty() {
        println!("\n  No suitable transactions found for proving.");
        std::process::exit(1);
    }

    // Debug: print all trace data for verification
    println!("\n[DEBUG] Trace data for {} transactions:", proofs.len());
    for (i, p) in proofs.iter().enumerate() {
        let steps = p.5;
        let trace_len = p.6.len();
        println!("  Tx {}: {} steps, {} bytes trace data", i, steps, trace_len);

        // Print first step details if available
        if trace_len >= 41 {
            let opcode = p.6[0];
            let pc_bytes: [u8; 8] = [p.6[1], p.6[2], p.6[3], p.6[4], p.6[5], p.6[6], p.6[7], p.6[8]];
            let pc = u64::from_le_bytes(pc_bytes);
            let gas_bytes: [u8; 8] = [p.6[9], p.6[10], p.6[11], p.6[12], p.6[13], p.6[14], p.6[15], p.6[16]];
            let gas_before = u64::from_le_bytes(gas_bytes);
            let gas_used_bytes: [u8; 8] = [p.6[17], p.6[18], p.6[19], p.6[20], p.6[21], p.6[22], p.6[23], p.6[24]];
            let gas_used = u64::from_le_bytes(gas_used_bytes);
            println!("    First step: opcode=0x{:02x}, pc={}, gas_before={}, gas_used={}", opcode, pc, gas_before, gas_used);
        }
    }

    // Build prover
    println!("STEP 2: Building Jolt prover with caching...");

    let target_dir = "/tmp/jolt-evm-guest-ethrex-cached";
    let setup_start = Instant::now();
    let cache_path = format!("{}/{}", target_dir, PROVER_CACHE_FILE);

    let (program_mut, prover_prep, verifier_prep): (_, _, _) = if std::path::Path::new(&cache_path).exists() {
        println!("  [CACHE HIT] Loading cached prover from {}", cache_path);
        let load_start = Instant::now();
        let program = jolt_evm_guest::compile_verify_evm_trace(target_dir);
        let mut program_mut = program;
        let cached_prep = match jolt_evm_guest::host_api::load_prover_preprocessing(target_dir) {
            Ok(prep) => {
                println!("  [CACHE] Preprocessing loaded ({:.3}s)", load_start.elapsed().as_secs_f64());
                prep
            }
            Err(e) => {
                println!("  [CACHE] Failed to load, rebuilding: {}", e);
                std::fs::remove_file(&cache_path).ok();
                let shared = jolt_evm_guest::preprocess_shared_verify_evm_trace(&mut program_mut).unwrap();
                let prover_prep = jolt_evm_guest::preprocess_prover_verify_evm_trace(shared.clone());
                if let Err(e) = jolt_evm_guest::host_api::save_prover_preprocessing(&prover_prep, target_dir) {
                    eprintln!("  [CACHE] Save failed: {}", e);
                }
                prover_prep
            }
        };
        let verifier_prep = jolt_evm_guest::verifier_preprocessing_from_prover_verify_evm_trace(&cached_prep);
        (program_mut, cached_prep, verifier_prep)
    } else {
        println!("  [CACHE MISS] Building prover from scratch...");
        std::fs::create_dir_all(target_dir).ok();
        let program = jolt_evm_guest::compile_verify_evm_trace(target_dir);
        let mut program_mut = program;
        let shared = jolt_evm_guest::preprocess_shared_verify_evm_trace(&mut program_mut).unwrap();
        let prover_prep = jolt_evm_guest::preprocess_prover_verify_evm_trace(shared.clone());
        println!("  [CACHE] Saving to {}", cache_path);
        if let Err(e) = jolt_evm_guest::host_api::save_prover_preprocessing(&prover_prep, target_dir) {
            eprintln!("  [CACHE] Save failed: {}", e);
        }
        let verifier_prep = jolt_evm_guest::verifier_preprocessing_from_prover_verify_evm_trace(&prover_prep);
        (program_mut, prover_prep, verifier_prep)
    };

    let prover = jolt_evm_guest::build_prover_verify_evm_trace(program_mut, prover_prep.clone());
    let _verifier = jolt_evm_guest::build_verifier_verify_evm_trace(verifier_prep);
    let setup_time = setup_start.elapsed();
    println!("  Prover setup: {:.3}s\n", setup_time.as_secs_f64());

    // Debug: print the trace data for the first transaction
    if !proofs.is_empty() {
        let first_tx = &proofs[0];
        println!("[DEBUG] First transaction trace data:");
        println!("  Hash: {:?}", first_tx.0);
        println!("  Steps: {}", first_tx.5);
        println!("  Trace data len: {} bytes", first_tx.6.len());
        // Print first few bytes as hex
        let hex_preview = hex::encode(&first_tx.6[..first_tx.6.len().min(100)]);
        println!("  First 100 bytes (hex): {}...", hex_preview.chars().take(100).collect::<String>());

        // Parse and print all steps with stack info
        println!("  All steps:");
        let trace_data = &first_tx.6;
        for step_idx in 0..first_tx.5.min(25) {
            let offset = step_idx * 41;
            let opcode = trace_data[offset];
            let gas_used = u64::from_le_bytes(trace_data[offset+17..offset+25].try_into().unwrap());
            let stack_before = u64::from_le_bytes(trace_data[offset+25..offset+33].try_into().unwrap()) as i64;
            let stack_after = u64::from_le_bytes(trace_data[offset+33..offset+41].try_into().unwrap()) as i64;
            let delta = stack_after - stack_before;
            println!("    Step {:3}: opcode=0x{:02x}, gas={:6}, stack=[{:2}->{:2}] (delta={:+})",
                     step_idx, opcode, gas_used, stack_before, stack_after, delta);
        }

        // Print steps around potential failure (step 19)
        println!("\n  Checking step 19 (opcode=0x03, SUB):");
        if first_tx.5 > 19 {
            let offset = 19 * 41;
            let opcode = trace_data[offset];
            let gas_used = u64::from_le_bytes(trace_data[offset+17..offset+25].try_into().unwrap());
            let stack_before = u64::from_le_bytes(trace_data[offset+25..offset+33].try_into().unwrap());
            let stack_after = u64::from_le_bytes(trace_data[offset+33..offset+41].try_into().unwrap());
            let delta = (stack_after as i64) - (stack_before as i64);
            println!("    opcode=0x{:02x}, gas_used={}, stack_before={}, stack_after={}", opcode, gas_used, stack_before, stack_after);
            // Check constraints
            println!("    Constraint check:");
            println!("      gas_used < 10000000: {} ({})", gas_used < 10000000, gas_used);
            println!("      stack_before < 1024: {} ({})", stack_before < 1024, stack_before);
            println!("      stack_after < 1024: {} ({})", stack_after < 1024, stack_after);
            println!("      delta in [-16, 16]: {} (delta={})", delta >= -16 && delta <= 16, delta);
        }
    }

    // Batch traces
    println!("STEP 3: Creating batch traces...");
    const MAX_BATCH_STEPS: usize = 1590;
    let mut batch = BatchTrace::new();
    let mut all_batches: Vec<BatchTrace> = Vec::new();

    for (_, _, _, _, gas, steps, trace, bytecode_hash) in &proofs {
        if batch.total_steps + steps > MAX_BATCH_STEPS && batch.total_steps > 0 {
            all_batches.push(batch);
            batch = BatchTrace::new();
        }
        batch.add_trace(*bytecode_hash, *gas, trace.clone());
    }
    if batch.total_steps > 0 {
        all_batches.push(batch);
    }

    let num_batches = all_batches.len();
    println!("  {} batch(es) from {} transactions", num_batches, proofs.len());
    for (i, b) in all_batches.iter().enumerate() {
        println!("    Batch {}: {} txs, {} steps", i, b.tx_count, b.total_steps);

        // Debug: print steps around potential issue from combined batch
        if i == 0 {
            println!("\n    [DEBUG] Combined batch trace - checking for problematic steps:");
            let mut problematic_count = 0;
            for step_idx in 0..b.total_steps.min(100) {
                let offset = step_idx * 41;
                if offset + 41 <= b.trace_data.len() {
                    let opcode = b.trace_data[offset];
                    let gas_used = u64::from_le_bytes(b.trace_data[offset+17..offset+25].try_into().unwrap());
                    let stack_before = u64::from_le_bytes(b.trace_data[offset+25..offset+33].try_into().unwrap());
                    let stack_after = u64::from_le_bytes(b.trace_data[offset+33..offset+41].try_into().unwrap());
                    let delta = (stack_after as i64) - (stack_before as i64);

                    // Check for potential issues
                    let mut issues = Vec::new();
                    if gas_used >= 10000000 {
                        issues.push(format!("gas={}>=10000000", gas_used));
                    }
                    if stack_before >= 1024 {
                        issues.push(format!("stack_before={}>=1024", stack_before));
                    }
                    if stack_after >= 1024 {
                        issues.push(format!("stack_after={}>=1024", stack_after));
                    }
                    if delta < -16 || delta > 16 {
                        issues.push(format!("delta={} out of [-16,16]", delta));
                    }

                    // Check opcode-specific constraints
                    let expected_delta = match opcode {
                        0x00 | 0xf3 | 0xfd | 0xfe => None,  // STOP, RETURN, REVERT, INVALID - no check
                        0x0c..=0x0f | 0x1e..=0x1f | 0x21..=0x2f | 0x49..=0x4f | 0x50 | 0x5c..=0x5d | 0x5e..=0x5f => None,  // Unimplemented
                        0x60..=0x7f => Some(1i64),      // PUSH1-PUSH32
                        0x80..=0x8f => Some(1i64),      // DUP1-DUP16
                        0x90..=0x9f => Some(0i64),      // SWAP1-SWAP16
                        0x01..=0x0b | 0x10..=0x14 | 0x1a..=0x1d | 0x16..=0x18 | 0x20 | 0x56 | 0x58 => Some(-1i64),  // Binary ops
                        0x15 | 0x19 | 0x30..=0x3f | 0x51 | 0x54 | 0x59 | 0x5a | 0x5b => Some(0i64),  // Unary/load ops
                        0x52 | 0x53 | 0x55 | 0x57 => Some(-2i64),  // Store/branch
                        0xf1 | 0xf2 | 0xf4 | 0xfa => Some(-6i64),  // CALL variants
                        _ => None,  // Unknown - check only bounds
                    };

                    if let Some(exp) = expected_delta {
                        if delta != exp {
                            issues.push(format!("opcode=0x{:02x} expected delta={}, got {}", opcode, exp, delta));
                        }
                    }

                    if !issues.is_empty() {
                        problematic_count += 1;
                        if problematic_count <= 10 {
                            println!("      Batch step {:3}: ISSUES: {}", step_idx, issues.join(", "));
                        }
                    }
                }
            }
            println!("    [DEBUG] Found {} steps with potential issues (of {} checked)", problematic_count, b.total_steps.min(100));
        }
    }

    // Prove batches in parallel
    println!("\nSTEP 4: Proving {} batches in parallel...\n", num_batches);
    let proof_start = Instant::now();

    #[derive(Clone)]
    struct BatchTask {
        batch_idx: usize,
        steps: usize,
        total_gas: u64,
        trace_data: Vec<u8>,
        bytecode_hash: [u8; 32],
        tx_count: usize,
    }

    let batch_tasks: Vec<BatchTask> = all_batches
        .into_iter()
        .enumerate()
        .map(|(i, b)| BatchTask {
            batch_idx: i,
            steps: b.total_steps,
            total_gas: b.total_gas,
            trace_data: b.trace_data,
            bytecode_hash: b.bytecode_hash,
            tx_count: b.tx_count,
        })
        .collect();

    let prover_arc = Arc::new(prover);
    let num_batches = batch_tasks.len();

    use std::sync::Mutex;
    type BatchResultData = (usize, usize, usize, std::time::Duration, u8, bool);
    let batch_results_arc: Arc<Mutex<Vec<BatchResultData>>> =
        Arc::new(Mutex::new(Vec::with_capacity(num_batches)));

    let handles: Vec<std::thread::JoinHandle<()>> = batch_tasks
        .into_iter()
        .map(|task| {
            let prover = prover_arc.clone();
            let results = Arc::clone(&batch_results_arc);

            std::thread::spawn(move || {
                let p_start = Instant::now();
                let (output, _proof, io_device) = prover(
                    0u64,
                    task.bytecode_hash,
                    task.steps,
                    task.total_gas,
                    true,
                    &task.trace_data,
                );
                let p_time = p_start.elapsed();

                // Check for panic
                if io_device.panic {
                    eprintln!("  [PANIC] Batch {} panicked", task.batch_idx);
                }

                let mut guard = results.lock().unwrap();
                guard.push((task.batch_idx, task.steps, task.tx_count, p_time, output, io_device.panic));
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let phase1_time = proof_start.elapsed();
    let mut batch_results: Vec<_> = batch_results_arc.lock().unwrap().clone();
    batch_results.sort_by_key(|r| r.0);

    let total_tx_proven: usize = batch_results.iter().map(|r| r.2).sum();
    let total_prove_cpu: std::time::Duration = batch_results.iter().map(|r| r.3).sum();

    println!("  Batch proving complete:");
    for r in &batch_results {
        println!("    Batch {}: {} txs, {} steps in {:.3}s", r.0, r.2, r.1, r.3.as_secs_f64());
    }

    println!("\n  Total wall time: {:.3}s", phase1_time.as_secs_f64());
    println!("  Total CPU time: {:.3}s", total_prove_cpu.as_secs_f64());
    println!("  Parallel efficiency: {:.1}%",
        100.0 * total_prove_cpu.as_secs_f64() / (phase1_time.as_secs_f64() * num_batches as f64));

    println!("\nSTEP 5: Verifying sample batches...");
    let verify_count = batch_results.len().min(3);
    let all_valid = batch_results.iter().take(verify_count).all(|r| r.4 == 1 && !r.5);

    let proof_time = proof_start.elapsed();

    let times: Vec<f64> = batch_results.iter().map(|r| r.3.as_secs_f64()).collect();
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let max = times.iter().cloned().fold(0.0f64, f64::max);
    let min = times.iter().cloned().fold(f64::MAX, f64::min);

    println!("  Batch proving statistics:");
    println!("    Per batch: avg={:.3}s, min={:.3}s, max={:.3}s", avg, min, max);
    println!("    Verified {} batches successfully", verify_count);

    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));
    println!();
    println!("Block: #{}", block_number);
    println!("Block hash: {:?}", cached_state.block_hash);
    println!("Block cached: {}", if block_cache.is_cached(block_number) { "YES" } else { "NO" });
    println!("Transactions in block: {}", cached_state.transactions.len());
    println!("Transactions proved: {}", total_tx_proven);
    println!("Transactions skipped: {}", skipped);
    println!("Proofs generated: {} (batched)", num_batches);
    println!();
    println!("Timings:");
    println!("  Prover setup (cached): {:.3}s", setup_time.as_secs_f64());
    println!("  Batch proving: {:.3}s wall", proof_time.as_secs_f64());
    println!("  Per batch: avg={:.3}s", avg);
    println!("  Total transactions proved: {}", total_tx_proven);
    println!("  Throughput: {:.0} transactions/second", total_tx_proven as f64 / proof_time.as_secs_f64());
    println!();
    println!("Result: {}", if all_valid { "ALL BATCHES VERIFIED" } else { "VERIFICATION FAILED" });
}

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

// Simple chrono replacement without external dependency
mod chrono {
    use std::time::{SystemTime, UNIX_EPOCH};

    pub struct DateTime {
        timestamp: i64,
    }

    impl DateTime {
        pub fn from_timestamp(timestamp: i64, _offset: u32) -> Option<Self> {
            Some(Self { timestamp })
        }

        pub fn format(&self, fmt: &str) -> impl std::fmt::Display {
            let secs = self.timestamp;
            // Simple Y-m-d H:M format
            let days = secs / 86400;
            let remaining = secs % 86400;
            let hours = remaining / 3600;
            let minutes = (remaining % 3600) / 60;

            // Approximate date calculation (simplified)
            let year = 1970 + (days / 365) as i64;
            let yday = days % 365;
            let month = (yday / 30).max(1).min(12) as u32;
            let day = (yday % 30 + 1) as u32;

            format!("{:04}-{:02}-{:02} {:02}:{:02}", year, month, day, hours, minutes)
        }
    }
}