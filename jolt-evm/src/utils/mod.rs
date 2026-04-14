//! Utility functions for jolt-evm
//!
//! Contains pure-Rust implementations of EVM crypto operations.

use crate::evm::H256;

/// Compute keccak256 hash
///
/// This is a pure Rust implementation suitable for no_std environments.
pub fn keccak256(data: &[u8]) -> H256 {
    use tiny_keccak::Hasher;
    let mut hasher = tiny_keccak::Keccak::v256();
    let mut output = [0u8; 32];
    hasher.update(data);
    hasher.finalize(&mut output);
    H256(output)
}

/// Compute keccak256 of multiple concatenated inputs
pub fn keccak256_concat(data: &[&[u8]]) -> H256 {
    use tiny_keccak::Hasher;
    let mut hasher = tiny_keccak::Keccak::v256();
    for d in data {
        hasher.update(d);
    }
    let mut output = [0u8; 32];
    hasher.finalize(&mut output);
    H256(output)
}
