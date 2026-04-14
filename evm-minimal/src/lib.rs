//! Minimal EVM implementation for Jolt zkVM
//!
//! This crate provides a minimal, no_std-compatible EVM implementation
//! that can be proven with Jolt's ZK proving system.
//!
//! # Architecture
//!
//! - **EVM Engine**: Uses LEVM from ethrex for bytecode execution
//! - **Types**: Minimal Address/H256/U256 using ruint
//! - **Crypto**: Pure Rust crypto (k256, ark-bn254) for no_std compatibility
//! - **Serialization**: postcard for input/output

#![no_std]

extern crate alloc;

mod types;
mod crypto;
mod input;
mod output;

pub use types::{Address, H256, U256};
pub use input::EvmInput;
pub use output::EvmOutput;
pub use crypto::{EvmCrypto, CryptoError, DefaultCrypto};

// Re-export LEVM types
pub use ethrex_levm::vm::VM;
pub use ethrex_levm::environment::Environment;
pub use ethrex_levm::errors::{VMError, InternalError, OpcodeResult, ExecutionReport};
pub use ethrex_levm::account::LevmAccount;
pub use ethrex_levm::db::gen_db::GeneralizedDatabase;

/// Compute keccak256 hash
pub fn keccak256(data: &[u8]) -> H256 {
    use tiny_keccak::Hasher;
    let mut hasher = tiny_keccak::Keccak::v256();
    let mut output = [0u8; 32];
    hasher.update(data);
    hasher.finalize(&mut output);
    H256(output)
}
