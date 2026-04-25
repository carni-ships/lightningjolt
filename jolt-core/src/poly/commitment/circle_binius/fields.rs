//! Binary tower field implementations for CircleBinius backend
//!
//! Binary fields F_{2^k} are represented as polynomials over GF(2).
//! Tower construction allows efficient arithmetic at each level.

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use num_traits::{One, Zero};

// =============================================================================
// F_{2^8} - 8-bit binary field
// =============================================================================

/// Element of F_{2^8} represented as u8.
/// Used for: opcodes, lookup table indices, small constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BinaryField8b(pub(crate) u8);

impl BinaryField8b {
    /// Create from raw integer representation
    pub fn from_repr(repr: u8) -> Self {
        BinaryField8b(repr)
    }

    /// Get raw integer representation
    pub fn to_repr(self) -> u8 {
        self.0
    }

    /// Additive identity (0)
    pub fn zero() -> Self {
        BinaryField8b(0)
    }

    /// Multiplicative identity (1)
    pub fn one() -> Self {
        BinaryField8b(1)
    }

    /// Multiplicative inverse using exponentiation
    /// a^(-1) = a^(2^8 - 2) = a^254 for F_{2^8}
    pub fn invert(self) -> Option<Self> {
        if self.0 == 0 {
            return None;
        }
        // Exponentiation by squaring: a^(254) = a^(11111110 binary)
        let mut result = Self::one();
        let mut base = self;
        // 254 = 0b11111110
        let exp = 254u8;
        let mut e = exp;
        while e > 0 {
            if e & 1 != 0 {
                result = result * base;
            }
            base = base.square();
            e >>= 1;
        }
        Some(result)
    }

    /// Square the field element
    /// In GF(2^8), a^2 can be computed efficiently using the Frobenius map
    pub fn square(self) -> Self {
        // Use multiply with self for correctness
        BinaryField8b(multiply_8(self.0, self.0))
    }
}

impl Add for BinaryField8b {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        BinaryField8b(self.0 ^ other.0)
    }
}

impl Sub for BinaryField8b {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        BinaryField8b(self.0 ^ other.0)
    }
}

impl Neg for BinaryField8b {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self
    }
}

// Multiplication for F_{2^8} using peasant's algorithm
// Reduction polynomial: x^8 + x^4 + x^3 + x + 1 (0x11B, but we use 0x1B since x^8 is implicit)
#[inline(always)]
fn multiply_8(a: u8, b: u8) -> u8 {
    let mut result = 0u8;
    let mut a = a;
    let mut b = b;
    for _ in 0..8 {
        if a & 1 != 0 {
            result ^= b;
        }
        let high_bit = b & 0x80;
        b <<= 1;
        if high_bit != 0 {
            b ^= 0x1B; // Reduce if x^8 term would be set
        }
        a >>= 1;
    }
    result
}

impl Mul for BinaryField8b {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        BinaryField8b(multiply_8(self.0, other.0))
    }
}

impl Zero for BinaryField8b {
    fn zero() -> Self {
        BinaryField8b(0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for BinaryField8b {
    fn one() -> Self {
        BinaryField8b(1)
    }
}

impl fmt::Display for BinaryField8b {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:02x}", self.0)
    }
}

// =============================================================================
// F_{2^{32}} - 32-bit binary field
// =============================================================================

/// Element of F_{2^{32}} represented as u32.
/// Used for: register values, memory words.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BinaryField32b(pub(crate) u32);

impl BinaryField32b {
    /// Create from raw integer representation
    pub fn from_repr(repr: u32) -> Self {
        BinaryField32b(repr)
    }

    /// Get raw integer representation
    pub fn to_repr(self) -> u32 {
        self.0
    }

    /// Additive identity (0)
    pub fn zero() -> Self {
        BinaryField32b(0)
    }

    /// Multiplicative identity (1)
    pub fn one() -> Self {
        BinaryField32b(1)
    }

    /// Multiplicative inverse using exponentiation
    /// a^(-1) = a^(2^32 - 2) for F_{2^32}
    pub fn invert(self) -> Option<Self> {
        if self.0 == 0 {
            return None;
        }
        // Use square-and-multiply for a^(2^32 - 2)
        // 2^32 - 2 = 0xFFFFFFFE (all bits set except bit 0)
        let mut result = Self::one();
        let mut base = self;
        // 2^32 - 2 has bits 31 down to 1 set, bit 0 is 0
        for _ in 0..32 {
            result = result * base;
            base = base.square();
        }
        // At this point result = a^33, but we need a^(2^32 - 2) = a^(4294967294)
        // Actually the loop above computed a^(2^32) since we did 32 squarings and 32 muls
        // We need to correct this: a^(2^32 - 2) = a^(2^32) / a^2 = a^(-2) mod p
        // But we can't divide. Instead let's recompute properly
        // Use binary exponentiation
        let exp = 0xFFFFFFFEu32;
        result = Self::one();
        base = self;
        let mut e = exp;
        while e > 0 {
            if e & 1 != 0 {
                result = result * base;
            }
            base = base.square();
            e >>= 1;
        }
        Some(result)
    }

    /// Square the field element
    pub fn square(self) -> Self {
        // Use multiply with self
        BinaryField32b(multiply_32(self.0, self.0))
    }
}

impl Add for BinaryField32b {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        BinaryField32b(self.0 ^ other.0)
    }
}

impl Sub for BinaryField32b {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        BinaryField32b(self.0 ^ other.0)
    }
}

impl Neg for BinaryField32b {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self
    }
}

// Multiplication for F_{2^32} using peasant's algorithm
// Reduction polynomial: x^32 + x^22 + x^2 + x + 1
#[inline(always)]
fn multiply_32(a: u32, b: u32) -> u32 {
    let mut result = 0u32;
    let mut a = a;
    let mut b = b;
    // Low 32 bits of reduction polynomial x^32 + x^22 + x^2 + x + 1
    let poly = 0x004000C5u32;
    for _ in 0..32 {
        if a & 1 != 0 {
            result ^= b;
        }
        let high_bit = b & 0x80000000;
        b <<= 1;
        if high_bit != 0 {
            b ^= poly;
        }
        a >>= 1;
    }
    result
}

impl Mul for BinaryField32b {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        BinaryField32b(multiply_32(self.0, other.0))
    }
}

impl Zero for BinaryField32b {
    fn zero() -> Self {
        BinaryField32b(0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for BinaryField32b {
    fn one() -> Self {
        BinaryField32b(1)
    }
}

impl fmt::Display for BinaryField32b {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:08x}", self.0)
    }
}

// =============================================================================
// F_{2^{64}} - 64-bit binary field
// =============================================================================

/// Element of F_{2^{64}} represented as u64.
/// Used for: memory addresses, 64-bit values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BinaryField64b(pub(crate) u64);

impl BinaryField64b {
    /// Create from raw integer representation
    pub fn from_repr(repr: u64) -> Self {
        BinaryField64b(repr)
    }

    /// Get raw integer representation
    pub fn to_repr(self) -> u64 {
        self.0
    }

    /// Additive identity (0)
    pub fn zero() -> Self {
        BinaryField64b(0)
    }

    /// Multiplicative identity (1)
    pub fn one() -> Self {
        BinaryField64b(1)
    }

    /// Multiplicative inverse using binary GCD algorithm
    /// For F_{2^64} with irreducible polynomial x^64 + x^4 + x^3 + x + 1
    pub fn invert(self) -> Option<Self> {
        if self.0 == 0 {
            return None;
        }
        // Binary extended GCD for field inversion
        // We want to find inv such that self * inv = 1 mod p(x)
        // p(x) = x^64 + x^4 + x^3 + x + 1
        let poly = 0x800000000000001Bu64; // x^64 + x^4 + x^3 + x + 1 (high bit implicit)

        let mut u = self.0;
        let mut v = poly;
        let mut s = 1u64;
        let mut t = 0u64;

        // Binary GCD
        while u != 0 && u != 1 {
            // Reduce u if even
            if u & 1 == 0 {
                u >>= 1;
                if s & 1 != 0 {
                    s = s ^ poly;
                }
                s >>= 1;
            } else {
                // If u and v are both odd, subtract
                let temp = u;
                u = v - u;
                v = temp;
                if s < t {
                    core::mem::swap(&mut s, &mut t);
                }
                s = s ^ t;
                t = temp ^ t ^ (temp - s);
            }
        }

        // u is now 1 (if invertible) or 0 (if not)
        if u == 0 {
            return None;
        }

        // s is the inverse modulo the polynomial
        Some(BinaryField64b(s))
    }

    /// Square the field element
    pub fn square(self) -> Self {
        // Use multiply with self
        self * self
    }
}

impl Add for BinaryField64b {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        BinaryField64b(self.0 ^ other.0)
    }
}

impl Sub for BinaryField64b {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        BinaryField64b(self.0 ^ other.0)
    }
}

impl Neg for BinaryField64b {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self
    }
}

impl Mul for BinaryField64b {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        // Use peasant's algorithm for multiplication
        let mut result = 0u64;
        let mut a = self.0;
        let mut b = other.0;
        // Low 64 bits of reduction polynomial x^64 + x^4 + x^3 + x + 1
        let poly = 0x000000000000001Bu64;
        for _ in 0..64 {
            if a & 1 != 0 {
                result ^= b;
            }
            let high_bit = b & 0x8000000000000000u64;
            b <<= 1;
            if high_bit != 0 {
                b ^= poly;
            }
            a >>= 1;
        }
        BinaryField64b(result)
    }
}

impl Zero for BinaryField64b {
    fn zero() -> Self {
        BinaryField64b(0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for BinaryField64b {
    fn one() -> Self {
        BinaryField64b(1)
    }
}

impl fmt::Display for BinaryField64b {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:016x}", self.0)
    }
}

// =============================================================================
// F_{2^{128}} - 128-bit binary field
// =============================================================================

/// Element of F_{2^{128}} represented as two u64s.
/// Used for: FRI challenges, security parameters, hash outputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BinaryField128b(pub(crate) u64, pub(crate) u64);

impl BinaryField128b {
    /// Create from raw integer representation
    pub fn from_repr(repr: (u64, u64)) -> Self {
        BinaryField128b(repr.0, repr.1)
    }

    /// Get raw integer representation
    pub fn to_repr(self) -> (u64, u64) {
        (self.0, self.1)
    }

    /// Additive identity (0)
    pub fn zero() -> Self {
        BinaryField128b(0, 0)
    }

    /// Multiplicative identity (1)
    pub fn one() -> Self {
        BinaryField128b(1, 0)
    }

    /// Multiplicative inverse
    pub fn invert(self) -> Option<Self> {
        if self.0 == 0 && self.1 == 0 {
            return None;
        }
        // Simplified: use binary exponentiation
        let mut result = Self::one();
        let mut base = self;
        for _ in 0..127 {
            if base.0 & 1 != 0 {
                result.0 ^= base.0;
                result.1 ^= base.1;
            }
            base = base.square();
        }
        Some(result)
    }

    /// Square the field element
    pub fn square(self) -> Self {
        let mut result = (0u64, 0u64);
        let (a0, a1) = (self.0, self.1);

        // Bit interleaving square
        for i in 0..64 {
            result.0 ^= ((a0 >> i) & 1) << (2 * i);
            result.0 ^= ((a0 >> (63 - i)) & 1) << (2 * i + 1);
            result.1 ^= ((a1 >> i) & 1) << (2 * i);
            result.1 ^= ((a1 >> (63 - i)) & 1) << (2 * i + 1);
        }

        // Apply reduction (simplified)
        if result.1 >> 63 != 0 {
            result.0 ^= 0x8000000000000001u64;
            result.1 ^= 0x87u64;
        }
        BinaryField128b(result.0, result.1)
    }
}

impl Add for BinaryField128b {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        BinaryField128b(self.0 ^ other.0, self.1 ^ other.1)
    }
}

impl Sub for BinaryField128b {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        BinaryField128b(self.0 ^ other.0, self.1 ^ other.1)
    }
}

impl Neg for BinaryField128b {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self
    }
}

impl Mul for BinaryField128b {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let mut result = (0u64, 0u64);
        let (a0, a1) = (self.0, self.1);
        let (b0, _b1) = (other.0, other.1);

        // Simplified multiplication
        for i in 0..64 {
            if (b0 >> i) & 1 != 0 {
                result.0 ^= a0 << i;
                result.1 ^= a1 << i;
            }
        }

        // Reduction (simplified)
        if result.1 >> 63 != 0 {
            result.0 ^= 0x8000000000000001u64;
            result.1 ^= 0x87u64;
        }

        BinaryField128b(result.0, result.1)
    }
}

impl Zero for BinaryField128b {
    fn zero() -> Self {
        BinaryField128b(0, 0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0 && self.1 == 0
    }
}

impl One for BinaryField128b {
    fn one() -> Self {
        BinaryField128b(1, 0)
    }
}

impl fmt::Display for BinaryField128b {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:016x}{:016x}", self.1, self.0)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f8_add() {
        let a = BinaryField8b(0b10101010);
        let b = BinaryField8b(0b11110000);
        let c = a + b;
        assert_eq!(c.0, 0b01011010);
    }

    #[test]
    fn test_f8_mul() {
        let a = BinaryField8b(3);
        let b = BinaryField8b(5);
        let c = a * b;
        assert_eq!(c.0, 15);
    }

    #[test]
    fn test_f8_invert() {
        let a = BinaryField8b(3);
        let a_inv = a.invert().unwrap();
        let identity = a * a_inv;
        assert_eq!(identity.0, 1);
    }

    #[test]
    fn test_f8_pow() {
        let a = BinaryField8b(2);
        let a_sq = a.square();
        assert_eq!(a_sq.0, 4);
    }

    #[test]
    fn test_f32_add() {
        let a = BinaryField32b(0x12345678);
        let b = BinaryField32b(0x87654321);
        let c = a + b;
        assert_eq!(c.0, 0x12345678 ^ 0x87654321);
    }

    #[test]
    fn test_f32_mul() {
        let a = BinaryField32b(2);
        let b = BinaryField32b(3);
        let c = a * b;
        assert_eq!(c.0, 6);
    }

    #[test]
    fn test_f64_add() {
        let a = BinaryField64b(0x123456789ABCDEF0);
        let b = BinaryField64b(0xFEDCBA9876543210);
        let c = a + b;
        assert_eq!(c.0, 0x123456789ABCDEF0 ^ 0xFEDCBA9876543210);
    }

    #[test]
    fn test_f128_add() {
        let a = BinaryField128b(0x12345678, 0x90ABCDEF);
        let b = BinaryField128b(0xFEDCBA09, 0x87654321);
        let c = a + b;
        assert_eq!(c.0, 0x12345678 ^ 0xFEDCBA09);
        assert_eq!(c.1, 0x90ABCDEF ^ 0x87654321);
    }

    #[test]
    fn test_f128_mul() {
        let a = BinaryField128b(1, 0);
        let b = BinaryField128b(2, 0);
        let c = a * b;
        assert_eq!(c.0, 2);
        assert_eq!(c.1, 0);
    }
}