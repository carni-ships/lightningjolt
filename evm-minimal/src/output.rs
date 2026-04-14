//! EVM Output serialization
//!
//! Output from EVM execution is serialized using postcard (no_std compatible).

use serde::{Deserialize, Serialize};
use alloc::vec::Vec;

/// Log entry produced by EVM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogOutput {
    pub address: [u8; 20],
    pub topics: Vec<[u8; 32]>,
    pub data: Vec<u8>,
}

/// EVM Output - result of execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvmOutput {
    /// Whether execution succeeded
    pub success: bool,
    /// Return data / output
    pub return_data: Vec<u8>,
    /// Gas used
    pub gas_used: u64,
    /// Logs produced
    pub logs: Vec<LogOutput>,
    /// Final state root (if applicable)
    pub state_root: [u8; 32],
}

impl EvmOutput {
    pub fn success(return_data: Vec<u8>, gas_used: u64, logs: Vec<LogOutput>) -> Self {
        EvmOutput {
            success: true,
            return_data,
            gas_used,
            logs,
            state_root: [0u8; 32],
        }
    }

    pub fn failure(return_data: Vec<u8>, gas_used: u64) -> Self {
        EvmOutput {
            success: false,
            return_data,
            gas_used,
            logs: Vec::new(),
            state_root: [0u8; 32],
        }
    }

    /// Serialize to bytes for Jolt output
    pub fn to_bytes(&self) -> Vec<u8> {
        postcard::to_allocvec(self).unwrap_or_default()
    }

    /// Deserialize from Jolt output
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        postcard::from_bytes(data).ok()
    }
}
