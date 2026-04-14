//! EVM Input serialization
//!
//! Input for the EVM is serialized using postcard (no_std compatible).

use crate::types::{Address, U256, EvmEnvironment};
use serde::{Deserialize, Serialize};
use alloc::vec::Vec;

/// EVM Input - what we serialize/deserialize for proving
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvmInput {
    /// Contract bytecode
    pub bytecode: Vec<u8>,
    /// Calldata
    pub calldata: Vec<u8>,
    /// Caller address
    pub caller: [u8; 20],
    /// Recipient address (or CREATE address)
    pub recipient: [u8; 20],
    /// Chain ID
    pub chain_id: u64,
    /// Gas limit
    pub gas_limit: u64,
    /// Value transferred
    pub value: [u8; 32],
}

impl EvmInput {
    pub fn new(
        bytecode: Vec<u8>,
        calldata: Vec<u8>,
        caller: Address,
        recipient: Address,
        chain_id: u64,
        gas_limit: u64,
        value: U256,
    ) -> Self {
        EvmInput {
            bytecode,
            calldata,
            caller: caller.0,
            recipient: recipient.0,
            chain_id,
            gas_limit,
            value: value.to_little_endian(),
        }
    }

    pub fn caller(&self) -> Address {
        Address(self.caller)
    }

    pub fn recipient(&self) -> Address {
        Address(self.recipient)
    }

    pub fn value(&self) -> U256 {
        U256::from_little_endian(&self.value)
    }

    pub fn to_env(&self) -> EvmEnvironment {
        EvmEnvironment::new(
            self.caller(),
            self.recipient(),
            self.chain_id,
            self.gas_limit,
            self.value(),
        )
    }

    /// Serialize to bytes for Jolt input
    pub fn to_bytes(&self) -> Vec<u8> {
        postcard::to_allocvec(self).unwrap_or_default()
    }

    /// Deserialize from Jolt input
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        postcard::from_bytes(data).ok()
    }
}
