//! Minimal EVM types
//!
//! These are minimal type definitions that wrap ruint types
//! and provide the interfaces needed for Jolt compatibility.

use core::fmt;
use ruint::aliases::U256 as U256Rust;

/// 20-byte Ethereum address
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Address(pub [u8; 20]);

impl Address {
    pub const ZERO: Address = Address([0u8; 20]);
    pub const MAX: Address = Address([0xffu8; 20]);

    pub fn from_slice(slice: &[u8]) -> Self {
        let mut arr = [0u8; 20];
        arr.copy_from_slice(&slice[..20]);
        Address(arr)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        &mut self.0
    }

    pub fn to_hex(&self) -> alloc::string::String {
        hex::encode(self.0)
    }
}

impl fmt::Debug for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{}", hex::encode(self.0))
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{}", hex::encode(self.0))
    }
}

/// 32-byte hash (used for code hashes, storage slots, etc)
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct H256(pub [u8; 32]);

impl H256 {
    pub const ZERO: H256 = H256([0u8; 32]);
    pub const MAX: H256 = H256([0xffu8; 32]);

    pub fn from_slice(slice: &[u8]) -> Self {
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&slice[..32]);
        H256(arr)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn to_hex(&self) -> alloc::string::String {
        hex::encode(self.0)
    }

    /// Convert to U256 (for arithmetic)
    pub fn to_uint(&self) -> U256 {
        U256::from_little_endian(&self.0)
    }

    /// Create from U256
    pub fn from_uint(u: &U256) -> Self {
        H256(u.to_little_endian())
    }
}

impl fmt::Debug for H256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "H256({})", hex::encode(self.0))
    }
}

impl fmt::Display for H256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{}", hex::encode(self.0))
    }
}

/// U256 wrapper around ruint
#[derive(Clone, Copy, Default)]
pub struct U256(pub U256Rust);

impl U256 {
    pub const ZERO: U256 = U256(U256Rust::ZERO);
    pub const ONE: U256 = U256(U256Rust::ONE);

    pub fn from_little_endian(data: &[u8; 32]) -> Self {
        U256(U256Rust::from_le_bytes(*data))
    }

    pub fn to_little_endian(&self) -> [u8; 32] {
        self.0.to_le_bytes()
    }

    pub fn from_big_endian(data: &[u8; 32]) -> Self {
        U256(U256Rust::from_be_bytes(*data))
    }

    pub fn to_big_endian(&self) -> [u8; 32] {
        self.0.to_be_bytes()
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    pub fn checked_add(self, other: Self) -> Option<Self> {
        self.0.checked_add(other.0).map(U256)
    }

    pub fn checked_sub(self, other: Self) -> Option<Self> {
        self.0.checked_sub(other.0).map(U256)
    }

    pub fn checked_mul(self, other: Self) -> Option<Self> {
        self.0.checked_mul(other.0).map(U256)
    }

    pub fn checked_div(self, other: Self) -> Option<Self> {
        if other.0.is_zero() {
            None
        } else {
            Some(U256(self.0 / other.0))
        }
    }

    pub fn wrapping_add(self, other: Self) -> Self {
        U256(self.0.wrapping_add(other.0))
    }

    pub fn wrapping_mul(self, other: Self) -> Self {
        U256(self.0.wrapping_mul(other.0))
    }

    pub fn saturating_add(self, other: Self) -> Self {
        U256(self.0.saturating_add(other.0))
    }

    pub fn saturating_sub(self, other: Self) -> Self {
        U256(self.0.saturating_sub(other.0))
    }

    pub fn lt(self, other: Self) -> bool {
        self.0 < other.0
    }

    pub fn le(self, other: Self) -> bool {
        self.0 <= other.0
    }

    pub fn gt(self, other: Self) -> bool {
        self.0 > other.0
    }

    pub fn ge(self, other: Self) -> bool {
        self.0 >= other.0
    }

    pub fn bit_and(self, other: Self) -> Self {
        U256(self.0 & other.0)
    }

    pub fn bit_or(self, other: Self) -> Self {
        U256(self.0 | other.0)
    }

    pub fn bit_xor(self, other: Self) -> Self {
        U256(self.0 ^ other.0)
    }

    pub fn bit_not(self) -> Self {
        U256(!self.0)
    }

    pub fn shl(self, shift: u32) -> Self {
        U256(self.0 << shift)
    }

    pub fn shr(self, shift: u32) -> Self {
        U256(self.0 >> shift)
    }

    pub fn leading_zeros(&self) -> usize {
        self.0.leading_zeros()
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.0.try_into().ok()
    }
}

impl PartialEq for U256 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for U256 {}

impl core::ops::Add for U256 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        U256(self.0 + other.0)
    }
}

impl core::ops::Sub for U256 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        U256(self.0 - other.0)
    }
}

impl core::ops::Mul for U256 {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        U256(self.0 * other.0)
    }
}

impl core::ops::Div for U256 {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        U256(self.0 / other.0)
    }
}

impl core::ops::BitAnd for U256 {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        U256(self.0 & other.0)
    }
}

impl core::ops::BitOr for U256 {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        U256(self.0 | other.0)
    }
}

impl core::ops::BitXor for U256 {
    type Output = Self;
    fn bitxor(self, other: Self) -> Self {
        U256(self.0 ^ other.0)
    }
}

impl fmt::Debug for U256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// EVM Environment / Context
#[derive(Clone, Debug)]
pub struct EvmEnvironment {
    pub caller: Address,
    pub recipient: Address,
    pub chain_id: u64,
    pub gas_limit: u64,
    pub value: U256,
    pub block_number: u64,
    pub timestamp: u64,
}

impl EvmEnvironment {
    pub fn new(
        caller: Address,
        recipient: Address,
        chain_id: u64,
        gas_limit: u64,
        value: U256,
    ) -> Self {
        EvmEnvironment {
            caller,
            recipient,
            chain_id,
            gas_limit,
            value,
            block_number: 0,
            timestamp: 0,
        }
    }
}
