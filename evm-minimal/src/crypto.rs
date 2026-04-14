//! Cryptographic operations for EVM
//!
//! This module provides the crypto primitives needed by the EVM
//! using no_std compatible implementations.

use crate::types::Address;

/// Crypto error type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CryptoError {
    InvalidSignature,
    RecoveryFailed,
    InvalidInput,
    VerificationFailed,
    Other(alloc::string::String),
}

impl From<alloc::string::String> for CryptoError {
    fn from(s: alloc::string::String) -> Self {
        CryptoError::Other(s)
    }
}

/// Trait for crypto operations needed by EVM
pub trait EvmCrypto {
    /// ECDSA signature recovery (ecrecover precompile)
    fn ecrecover(&self, sig: &[u8; 64], recov: u8, msg: &[u8; 32]) -> Result<Address, CryptoError>;

    /// BN254 G1 scalar multiplication
    fn bn254_mul(&self, point: &[u8], scalar: &[u8]) -> Result<[u8; 64], CryptoError>;

    /// BN254 pairing check
    fn bn254_pairing(&self, pairs: &[( &[u8], &[u8])]) -> Result<bool, CryptoError>;
}

/// Keccak256 hash function
pub fn keccak256(data: &[u8]) -> [u8; 32] {
    use tiny_keccak::Hasher;
    let mut hasher = tiny_keccak::Keccak::v256();
    let mut output = [0u8; 32];
    hasher.update(data);
    hasher.finalize(&mut output);
    output
}

/// Default crypto implementation using k256
pub struct DefaultCrypto;

impl EvmCrypto for DefaultCrypto {
    fn ecrecover(&self, sig: &[u8; 64], recov: u8, msg: &[u8; 32]) -> Result<Address, CryptoError> {
        use k256::ecdsa::{RecoveryId, Signature, VerifyingKey};

        // Parse signature
        let sig_obj = Signature::from_slice(sig)
            .map_err(|_| CryptoError::InvalidSignature)?;

        // Handle canonical low-S normalization
        let mut recid_byte = recov;
        if let Some(_low_s) = sig_obj.normalize_s() {
            recid_byte ^= 1;
        }

        // Parse recovery ID
        let recovery_id = RecoveryId::from_byte(recid_byte)
            .ok_or(CryptoError::RecoveryFailed)?;

        // Recover public key
        let vk = VerifyingKey::recover_from_prehash(msg, &sig_obj, recovery_id)
            .map_err(|_| CryptoError::RecoveryFailed)?;

        // Get SEC1 uncompressed public key
        let uncompressed = vk.to_encoded_point(false);
        let uncompressed_bytes = uncompressed.as_bytes();

        // Hash the public key coordinates (excluding 0x04 prefix)
        let hash = keccak256(&uncompressed_bytes[1..65]);

        // Return address (last 20 bytes of hash)
        Ok(Address::from_slice(&hash[12..]))
    }

    fn bn254_mul(&self, _point: &[u8], _scalar: &[u8]) -> Result<[u8; 64], CryptoError> {
        // BN254 multiplication not implemented in default crypto
        // For now, return zeros - this would need substrate-bn or similar
        Err(CryptoError::Other("BN254 mul not implemented".into()))
    }

    fn bn254_pairing(&self, _pairs: &[( &[u8], &[u8])]) -> Result<bool, CryptoError> {
        // BN254 pairing not implemented in default crypto
        Err(CryptoError::Other("BN254 pairing not implemented".into()))
    }
}
