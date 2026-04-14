//! Core EVM types for jolt-evm
//!
//! These types are designed for no_std compatibility and ZK-proof constraints.
//! All data structures use plain old data (POD) for efficient serialization
//! and constraint checking.

/// EVM address (20 bytes)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Address(pub [u8; 20]);

/// EVM word (32 bytes - H256)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct H256(pub [u8; 32]);

impl H256 {
    /// Returns the zero hash
    pub fn zero() -> Self {
        Self([0u8; 32])
    }

    /// Check if hash is zero
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }
}

/// EVM unsigned 256-bit integer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct U256(pub [u8; 32]);

impl U256 {
    pub fn zero() -> Self {
        Self([0u8; 32])
    }

    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }
}

/// EVM opcode numbers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    // Arithmetic
    STOP = 0x00,
    ADD = 0x01,
    MUL = 0x02,
    SUB = 0x03,
    DIV = 0x04,
    SDIV = 0x05,
    MOD = 0x06,
    SMOD = 0x07,
    ADDMOD = 0x08,
    MULMOD = 0x09,
    EXP = 0x0a,
    SIGNEXTEND = 0x0b,

    // Comparison & Bitwise
    LT = 0x10,
    GT = 0x11,
    SLT = 0x12,
    SGT = 0x13,
    EQ = 0x14,
    ISZERO = 0x15,
    AND = 0x16,
    OR = 0x17,
    XOR = 0x18,
    NOT = 0x19,
    BYTE = 0x1a,
    SHL = 0x1b,
    SHR = 0x1c,
    SAR = 0x1d,

    // SHA3
    SHA3 = 0x20,

    // Memory
    MLOAD = 0x51,
    MSTORE = 0x52,
    MSTORE8 = 0x53,

    // Storage
    SLOAD = 0x54,
    SSTORE = 0x55,

    // Control Flow
    JUMP = 0x56,
    JUMPI = 0x57,
    PC = 0x58,
    JUMPDEST = 0x5b,

    // Stack
    PUSH1 = 0x60,
    PUSH2 = 0x61,
    PUSH3 = 0x62,
    PUSH4 = 0x63,
    DUP1 = 0x80,
    DUP2 = 0x81,
    SWAP1 = 0x90,
    SWAP2 = 0x91,

    // Return data
    RETURNDATASIZE = 0x3d,
    RETURNDATACOPY = 0x3e,

    // Logs
    LOG0 = 0xa0,
    LOG1 = 0xa1,
    LOG2 = 0xa2,
    LOG3 = 0xa3,
    LOG4 = 0xa4,

    // System
    CREATE = 0xf0,
    CALL = 0xf1,
    CALLCODE = 0xf2,
    RETURN = 0xf3,
    DELEGATECALL = 0xf4,
    CREATE2 = 0xf5,
    STATICCALL = 0xfa,
    REVERT = 0xfd,
    INVALID = 0xfe,
    SELFDBALANCE = 0xf7,
    SELFDESTRUCT = 0xff,
}

impl Opcode {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x00 => Some(Opcode::STOP),
            0x01 => Some(Opcode::ADD),
            0x02 => Some(Opcode::MUL),
            0x03 => Some(Opcode::SUB),
            0x04 => Some(Opcode::DIV),
            0x05 => Some(Opcode::SDIV),
            0x06 => Some(Opcode::MOD),
            0x07 => Some(Opcode::SMOD),
            0x08 => Some(Opcode::ADDMOD),
            0x09 => Some(Opcode::MULMOD),
            0x0a => Some(Opcode::EXP),
            0x0b => Some(Opcode::SIGNEXTEND),
            0x10 => Some(Opcode::LT),
            0x11 => Some(Opcode::GT),
            0x12 => Some(Opcode::SLT),
            0x13 => Some(Opcode::SGT),
            0x14 => Some(Opcode::EQ),
            0x15 => Some(Opcode::ISZERO),
            0x16 => Some(Opcode::AND),
            0x17 => Some(Opcode::OR),
            0x18 => Some(Opcode::XOR),
            0x19 => Some(Opcode::NOT),
            0x1a => Some(Opcode::BYTE),
            0x1b => Some(Opcode::SHL),
            0x1c => Some(Opcode::SHR),
            0x1d => Some(Opcode::SAR),
            0x20 => Some(Opcode::SHA3),
            0x51 => Some(Opcode::MLOAD),
            0x52 => Some(Opcode::MSTORE),
            0x53 => Some(Opcode::MSTORE8),
            0x54 => Some(Opcode::SLOAD),
            0x55 => Some(Opcode::SSTORE),
            0x56 => Some(Opcode::JUMP),
            0x57 => Some(Opcode::JUMPI),
            0x58 => Some(Opcode::PC),
            0x5b => Some(Opcode::JUMPDEST),
            0x60..=0x7f => Some(Opcode::PUSH1), // PUSH1-PUSH32 mapped to PUSH1
            0x80..=0x8f => Some(Opcode::DUP1), // DUP1-DUP16 mapped to DUP1
            0x90..=0x9f => Some(Opcode::SWAP1), // SWAP1-SWAP16 mapped to SWAP1
            0xa0..=0xa4 => Some(Opcode::LOG0), // LOG0-LOG4 mapped to LOG0
            0xf0 => Some(Opcode::CREATE),
            0xf1 => Some(Opcode::CALL),
            0xf2 => Some(Opcode::CALLCODE),
            0xf3 => Some(Opcode::RETURN),
            0xf4 => Some(Opcode::DELEGATECALL),
            0xf5 => Some(Opcode::CREATE2),
            0xfa => Some(Opcode::STATICCALL),
            0xfd => Some(Opcode::REVERT),
            0xfe => Some(Opcode::INVALID),
            0xf7 => Some(Opcode::SELFDBALANCE),
            0xff => Some(Opcode::SELFDBALANCE), // SELFDESTRUCT mapped to SELFDBALANCE
            _ => None,
        }
    }
}

/// A single EVM execution step
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EvmStep {
    /// Program counter
    pub pc: u64,
    /// Opcode being executed
    pub opcode: u8,
    /// Gas remaining before this step
    pub gas: u64,
    /// Gas used by this step
    pub gas_used: u64,
    /// Stack state after this step (as list of U256 words)
    pub stack: alloc::vec::Vec<U256>,
    /// Memory state after this step (offset + data)
    pub memory: Option<(u64, alloc::vec::Vec<u8>)>,
    /// Storage key affected (if any)
    pub storage_key: Option<H256>,
    /// Storage value before
    pub storage_value_before: Option<H256>,
    /// Storage value after
    pub storage_value_after: Option<H256>,
    /// Return data offset + size (for RETURNDATACOPY, CALL, etc.)
    pub return_data: Option<(u64, u64)>,
}

/// Complete EVM execution trace
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EvmTrace {
    /// Initial program counter
    pub entry_pc: u64,
    /// Contract code hash
    pub code_hash: H256,
    /// Caller address
    pub caller: Address,
    /// Callee address
    pub callee: Address,
    /// Call value
    pub value: U256,
    /// Input data
    pub input_data: alloc::vec::Vec<u8>,
    /// All execution steps
    pub steps: alloc::vec::Vec<EvmStep>,
    /// Final gas remaining
    pub final_gas: u64,
    /// Whether execution was successful
    pub success: bool,
    /// Return data (if successful)
    pub return_data: alloc::vec::Vec<u8>,
}

/// EVM state snapshot at a checkpoint
#[derive(Debug, Clone)]
pub struct EvmState {
    /// Program counter
    pub pc: u64,
    /// Gas remaining
    pub gas: u64,
    /// Stack pointer (index into stack)
    pub sp: usize,
    /// Memory size (in bytes)
    pub memory_size: u64,
}

/// Stack frame for EVM calls
#[derive(Debug, Clone)]
pub struct StackFrame {
    /// Return PC
    pub return_pc: u64,
    /// Stack pointer at call
    pub sp: usize,
    /// Memory size at call
    pub memory_size: u64,
    /// Gas at call
    pub gas: u64,
}
