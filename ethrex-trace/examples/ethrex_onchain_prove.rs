//! Execute a realistic Ethereum transaction and generate traces for Jolt proving.
//!
//! This example demonstrates:
//! 1. Setting up realistic EVM state (contracts, accounts)
//! 2. Creating a transaction that calls a Solidity contract
//! 3. Executing with step-by-step tracing via ethrex VM
//! 4. Generating 41-byte/step traces for Jolt verification
//!
//! Run with: cargo run -p ethrex-trace --example ethrex_onchain_prove

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

const STEP_SIZE: usize = 41;

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

/// In-memory database with pre-set state for realistic EVM execution
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

    /// Set up a simple storage contract at an address
    /// This mimics a deployed Solidity contract
    fn setup_storage_contract(&self, address: Address) {
        // Simple bytecode: SSTORE then SLOAD then STOP
        let code = Bytes::from(vec![
            0x60, 0x00, 0x60, 0x00, 0x55, // SSTORE
            0x60, 0x00, 0x60, 0x00, 0x54, // SLOAD
            0x00, // STOP
        ]);

        let code_hash = H256::from_str("0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef").unwrap();
        let code_obj = Code::from_bytecode(code.clone(), &NativeCrypto);
        self.code.write().unwrap().insert(code_hash, code_obj.clone());
        self.code_metadata.write().unwrap().insert(code_hash, CodeMetadata { length: code_obj.bytecode.len() as u64 });

        let mut accounts = self.accounts.write().unwrap();
        accounts.insert(address, AccountState {
            nonce: 1,
            balance: U256::from(1_000_000_000_000_000_000u64), // 1 ETH
            code_hash,
            storage_root: H256::zero(),
        });
    }

    /// Set up a funded account
    fn setup_account(&self, address: Address, balance: U256, nonce: u64) {
        let mut accounts = self.accounts.write().unwrap();
        accounts.insert(address, AccountState {
            nonce,
            balance,
            code_hash: H256::zero(),
            storage_root: H256::zero(),
        });
    }

    /// Pre-set a storage value
    fn set_storage(&self, address: Address, slot: H256, value: U256) {
        let mut storage = self.storage.write().unwrap();
        storage.insert((address, slot), value);
    }

    /// Set block hash for BLOCKHASH opcode
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

/// Create a simple contract call transaction
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
    println!("=== Realistic Ethereum Transaction Proving ===\n");

    // ============================================
    // SETUP: Create realistic EVM state
    // ============================================
    println!("Setting up EVM state...");

    let db = RealisticDatabase::new();

    // Set up a funded sender account
    let sender = Address::from_str("0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266").unwrap();
    db.setup_account(sender, U256::from(10u64.pow(18)), 0); // 10 ETH

    // Set up a contract at a known address
    let contract = Address::from_str("0x1234567890123456789012345678901234567890").unwrap();
    db.setup_storage_contract(contract);

    // Pre-set a storage value (mimics a previously set contract state)
    let slot = H256::zero();
    db.set_storage(contract, slot, U256::from(42u64));

    // Set some block hashes (for BLOCKHASH opcode)
    for i in 0..256u64 {
        let hash = H256::from_str(&format!("0x{:064x}", i)).unwrap();
        db.set_block_hash(i, hash);
    }

    println!("  Sender: {}", sender);
    println!("  Contract: {}", contract);
    println!("  Pre-set storage[0] = 42");
    println!("\n");

    // ============================================
    // TRANSACTION: Call the contract
    // ============================================
    println!("Creating transaction...");

    // Create a transaction that calls the contract's store function
    // Function selector for "store(uint256)" = keccak256("store(uint256)")[0:4]
    // store(uint256) = 0x6057361d
    let input = Bytes::from(vec![0x60, 0x57, 0x36, 0x1d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2a]); // store(42)

    let tx = create_call_tx(
        sender,
        contract,
        U256::zero(),
        1_000_000,
        input,
    );

    println!("  From: {}", sender);
    println!("  To: {}", contract);
    println!("  Data: {:02x?}", tx.data());
    println!("\n");

    // ============================================
    // EXECUTE: Run through ethrex VM with tracing
    // ============================================
    println!("Executing transaction through ethrex VM...");

    // Wrap in GeneralizedDatabase
    let db = Arc::new(db);
    let mut gen_db = GeneralizedDatabase::new(db);

    // Build block header for environment (use defaults for fields we don't care about)
    let block_header = ethrex_common::types::BlockHeader {
        difficulty: U256::from(2u64.pow(31)),
        number: 1,
        gas_limit: 30_000_000,
        timestamp: 1_600_000_000,
        prev_randao: H256::zero(),
        base_fee_per_gas: Some(10u64.pow(9)),
        ..Default::default()
    };

    // Build environment
    let chain_config = gen_db.store.get_chain_config().unwrap();
    let config = EVMConfig::new_from_chain_config(&chain_config, &block_header);

    let env = ethrex_levm::environment::Environment {
        origin: sender,
        gas_limit: tx.gas_limit(),
        config,
        block_number: block_header.number,
        coinbase: block_header.coinbase,
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

    // Create VM
    let mut vm = VM::new(
        env,
        &mut gen_db,
        &tx,
        LevmCallTracer::disabled(),
        VMType::L1,
        &NativeCrypto,
    ).unwrap();

    // Set up trace generator
    let mut tracer = TraceGenerator::new();
    vm.set_step_tracer(&mut tracer);

    // Execute
    let start = Instant::now();
    let result = vm.execute();
    let elapsed = start.elapsed();

    match result {
        Ok(report) => {
            println!("\nExecution completed in {:?}", elapsed);
            println!("Success: {}", report.is_success());
            println!("Gas used: {}", report.gas_used);

            let trace_data = tracer.into_bytes();
            let steps = trace_data.len() / STEP_SIZE;
            println!("\n=== Trace Output ===");
            println!("Steps: {}", steps);
            println!("Trace bytes: {} ({} steps x {} bytes)", trace_data.len(), steps, STEP_SIZE);

            // Show first few steps
            println!("\nFirst 10 steps:");
            for i in 0..steps.min(10) {
                let offset = i * STEP_SIZE;
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
                    0x49 => "BLOBBASEFEE",
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
                    0x60..=0x7f => "PUSH",
                    0x80..=0x8f => "DUP",
                    0x90..=0x9f => "SWAP",
                    0xa0..=0xa4 => "LOG",
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

                println!("  {:2}: {:10} pc={:4} gas={:8} stack[{} -> {}]",
                    i, opcode_name, pc, gas_used, stack_before, stack_after);
            }

            if steps > 10 {
                println!("  ... ({} more steps)", steps - 10);
            }

            println!("\n=== Ready for Jolt Proving ===");
            println!("The trace_data ({} bytes) can now be fed to the Jolt prover/verifier.", trace_data.len());

        }
        Err(e) => {
            println!("Execution failed: {:?}", e);
        }
    }
}
