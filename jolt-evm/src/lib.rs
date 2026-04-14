//! Jolt-EVM: A zkEVM built on Jolt
//!
//! This crate implements a Type B1 zkEVM where:
//! - **Host** executes EVM bytecode and generates execution traces
//! - **Guest** verifies each step of execution using Jolt's constraint system
//!
//! # Architecture
//!
//! The guest program receives an execution trace from the host and verifies
//! each EVM opcode execution using `check_advice!` assertions. This approach
//! (B1 - full verification) provides the highest security guarantees.
//!
//! # Trace Format
//!
//! Each step in the trace contains:
//! - Opcode being executed
//! - Stack state (before/after)
//! - Memory state (before/after for memory operations)
//! - Storage state (before/after for SLOAD/SSTORE)
//! - Gas state
//! - PC and other execution context

#![no_std]

extern crate alloc;

mod evm;
pub mod opcodes;
mod utils;

pub use evm::{Address, H256, U256, EvmStep, EvmTrace, EvmState, Opcode};
pub use opcodes::{PreState, PostState, VerificationError, verify_step};
pub use utils::keccak256;

/// Maximum trace length for a single EVM execution
pub const MAX_TRACE_LENGTH: usize = 1 << 20; // 1M steps

/// Maximum stack size
pub const MAX_STACK_SIZE: usize = 1024;

/// Maximum memory size (in bytes)
pub const MAX_MEMORY_SIZE: usize = 1 << 26; // 64 MB

/// Maximum storage size (in slots)
pub const MAX_STORAGE_SIZE: usize = 1 << 20; // 1M slots
