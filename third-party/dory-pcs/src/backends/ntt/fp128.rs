//! 128-bit Field Implementation for Lattice-Based Dory
//!
//! Field: q = 2^64 - 2^32 + 1 (ML-KEM modulus)

use crate::primitives::lattice_trait::LatticeField;
use crate::primitives::serialization::{Compress, DoryDeserialize, DorySerialize, Valid};
use std::io::{Read, Write};

/// ML-KEM field element
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Fp128Element(pub u64);

impl Valid for Fp128Element {
    fn check(&self) -> Result<(), crate::primitives::serialization::SerializationError> {
        Ok(())
    }
}

impl DorySerialize for Fp128Element {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), crate::primitives::serialization::SerializationError> {
        writer.write_all(&self.0.to_le_bytes())?;
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        8
    }
}

impl DoryDeserialize for Fp128Element {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        _validate: crate::primitives::serialization::Validate,
    ) -> Result<Self, crate::primitives::serialization::SerializationError> {
        let mut bytes = [0u8; 8];
        reader.read_exact(&mut bytes)?;
        Ok(Self(u64::from_le_bytes(bytes)))
    }
}

impl Fp128Element {
    /// ML-KEM modulus: q = 2^64 - 2^32 + 1
    pub const MODULUS: u64 = 0xFFFFFFFF00000001u64;

    pub const fn from_raw(val: u64) -> Self {
        Self(val % Self::MODULUS)
    }

    pub const fn zero() -> Self {
        Self(0)
    }

    pub const fn one() -> Self {
        Self(1)
    }

    pub const fn is_zero(&self) -> bool {
        self.0 == 0
    }

    pub const fn is_one(&self) -> bool {
        self.0 == 1
    }

    /// Addition mod q
    #[inline(always)]
    pub fn add(&self, rhs: &Self) -> Self {
        let sum = self.0.wrapping_add(rhs.0);
        if sum >= Self::MODULUS || sum < self.0 {
            Self(sum.wrapping_sub(Self::MODULUS))
        } else {
            Self(sum)
        }
    }

    /// Subtraction mod q using 128-bit arithmetic
    #[inline(always)]
    pub fn sub(&self, rhs: &Self) -> Self {
        // Compute (self - rhs) mod q
        // Since a, b < q, we can compute a + q - b which is in [0, 2q)
        // If result >= q, subtract q to get into [0, q)
        let a = self.0 as u128;
        let b = rhs.0 as u128;
        let q = Self::MODULUS as u128;

        let diff = a + q - b;
        if diff >= q {
            Self((diff - q) as u64)
        } else {
            Self(diff as u64)
        }
    }

    /// Negation mod q
    #[inline(always)]
    pub fn neg(&self) -> Self {
        if self.0 == 0 {
            Self::zero()
        } else {
            Self(Self::MODULUS - self.0)
        }
    }

    /// Multiplication mod q using Barrett reduction
    pub fn mul(&self, rhs: &Self) -> Self {
        let (lo, hi) = Self::wideumul(self.0, rhs.0);
        Self::barrett_reduce(lo, hi)
    }

    /// Wide multiplication
    #[inline(always)]
    fn wideumul(a: u64, b: u64) -> (u64, u64) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::_mulx_u64;
            unsafe { _mulx_u64(a, b) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let result = (a as u128) * (b as u128);
            (result as u64, (result >> 64) as u64)
        }
    }

    /// Barrett reduction for q = 2^64 - 2^32 + 1
    fn barrett_reduce(lo: u64, hi: u64) -> Self {
        // Use 128-bit arithmetic to compute x mod q correctly.
        // x = hi*2^64 + lo, and we need x mod q.
        // Since q = 2^64 - 2^32 + 1, we have 2^64 ≡ 2^32 - 1 (mod q)
        // So x ≡ hi*(2^32 - 1) + lo (mod q) = hi*2^32 - hi + lo
        //
        // Let hi = hi_hi*2^32 + hi_lo, then:
        // hi*2^32 = hi_hi*2^64 + hi_lo*2^32
        // x = hi_hi*2^64 + hi_lo*2^32 - hi + lo
        // Since 2^64 ≡ 0 (mod q), hi_hi*2^64 ≡ 0
        // x ≡ hi_lo*2^32 - hi + lo = lo + hi_lo*2^32 - hi
        //
        // Now hi_lo*2^32 - hi can be in range [-q, q) roughly.
        // But hi_lo*2^32 - hi could be negative or larger than q.
        //
        // Use proper Barrett reduction: compute x mod q using 128-bit arithmetic
        let q = Self::MODULUS as u128;
        let x = ((hi as u128) << 64) | (lo as u128);
        let t = (x % q) as u64;
        Self(t)
    }

    pub fn inv(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        // Fermat's little theorem: a^(q-2) mod q
        let exp = Self::MODULUS - 2;
        Some(self.pow(exp))
    }

    /// Exponentiation by squaring
    pub fn pow(&self, mut exp: u64) -> Self {
        let mut base = *self;
        let mut result = Self::one();

        while exp > 0 {
            if exp & 1 != 0 {
                result = result * base;
            }
            base = base * base;
            exp >>= 1;
        }
        result
    }

    pub fn primitive_root_of_unity(log_n: usize) -> Self {
        // For ML-KEM field q = 2^64 - 2^32 + 1, we need omega with order 2^log_n.
        // Since q-1 = 2^32 * (2^32 - 1), there exist 2^32-th roots of unity.
        //
        // For omega = base^((q-1)/n) to be a primitive n-th root:
        // - omega^n = base^(q-1) = 1 (always true if base is in the field)
        // - omega^(n/2) != 1 (meaning omega has order exactly n, not a divisor)
        //
        // omega^(n/2) = base^((q-1)/2) = 1 means base is a quadratic residue.
        // omega^(n/2) = base^((q-1)/2) = -1 means base is a quadratic non-residue,
        // which is what we need for a primitive n-th root.

        let q_minus_1 = Self::MODULUS - 1;
        let n = 1u64 << log_n;  // 2^log_n
        let exponent = q_minus_1 / n;

        // Try bases until we find one that gives a primitive root
        // A primitive root has omega^(n/2) != 1 (specifically = -1 for n > 2)
        let candidate_bases = [3u64, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73];

        for &base in &candidate_bases {
            let omega = Self(base).pow(exponent);

            // omega = 1 has order 1, which is never what we want for n > 1
            if omega.is_one() {
                continue;
            }

            let omega_half = omega.pow(n / 2);

            // omega^(n/2) = 1 means omega has order dividing n/2 (not primitive)
            // omega^(n/2) = -1 means omega has order exactly n (IS primitive)
            // For n=4, we specifically need omega^2 = -1
            if omega_half.is_one() {
                continue; // omega has order 1 or 2, not n
            }

            // Verify omega^n = 1 (it should always, but let's double-check)
            if omega.pow(n).is_one() {
                return omega; // This is a valid primitive n-th root
            }
        }

        // This should never happen if the field is properly constructed
        // But if it does, we have a fundamental issue
        panic!("Failed to find primitive {}-th root of unity", n);
    }
}

impl std::ops::Add for Fp128Element {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Fp128Element::add(&self, &rhs)
    }
}

impl std::ops::Sub for Fp128Element {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Fp128Element::sub(&self, &rhs)
    }
}

impl std::ops::Mul for Fp128Element {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Fp128Element::mul(&self, &rhs)
    }
}

impl std::ops::Neg for Fp128Element {
    type Output = Self;
    fn neg(self) -> Self {
        Fp128Element::neg(&self)
    }
}

impl<'a> std::ops::Add<&'a Fp128Element> for Fp128Element {
    type Output = Self;
    fn add(self, rhs: &'a Fp128Element) -> Self {
        Fp128Element::add(&self, rhs)
    }
}

impl<'a> std::ops::Sub<&'a Fp128Element> for Fp128Element {
    type Output = Self;
    fn sub(self, rhs: &'a Fp128Element) -> Self {
        Fp128Element::sub(&self, rhs)
    }
}

impl<'a> std::ops::Mul<&'a Fp128Element> for Fp128Element {
    type Output = Self;
    fn mul(self, rhs: &'a Fp128Element) -> Self {
        Fp128Element::mul(&self, rhs)
    }
}

impl std::ops::AddAssign for Fp128Element {
    fn add_assign(&mut self, rhs: Self) {
        *self = Fp128Element::add(self, &rhs);
    }
}

impl std::ops::SubAssign for Fp128Element {
    fn sub_assign(&mut self, rhs: Self) {
        *self = Fp128Element::sub(self, &rhs);
    }
}

impl std::ops::MulAssign for Fp128Element {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Fp128Element::mul(self, &rhs);
    }
}

impl LatticeField for Fp128Element {
    fn zero() -> Self {
        Fp128Element::zero()
    }

    fn one() -> Self {
        Fp128Element::one()
    }

    fn is_zero(&self) -> bool {
        Fp128Element::is_zero(self)
    }

    fn add(&self, rhs: &Self) -> Self {
        *self + *rhs
    }

    fn sub(&self, rhs: &Self) -> Self {
        *self - *rhs
    }

    fn mul(&self, rhs: &Self) -> Self {
        *self * *rhs
    }

    fn inv(self) -> Option<Self> {
        Fp128Element::inv(&self)
    }

    fn random() -> Self {
        Fp128Element::from_raw(42)
    }

    fn from_u64(val: u64) -> Self {
        Fp128Element::from_raw(val)
    }

    fn from_i64(val: i64) -> Self {
        if val >= 0 {
            Self::from_u64(val as u64)
        } else {
            let abs = (-val) as u64;
            let val_mod = abs % Self::MODULUS;
            if val_mod == 0 {
                Self::zero()
            } else {
                Self(Self::MODULUS - val_mod)
            }
        }
    }
}

impl std::fmt::Debug for Fp128Element {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fp128({})", self.0)
    }
}

impl std::fmt::Display for Fp128Element {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for Fp128Element {
    fn from(val: u64) -> Self {
        Self::from_raw(val)
    }
}

impl From<u32> for Fp128Element {
    fn from(val: u32) -> Self {
        Self(val as u64)
    }
}

impl From<i64> for Fp128Element {
    fn from(val: i64) -> Self {
        Self::from_i64(val)
    }
}

#[derive(Clone, Default)]
pub struct Fp128Field;

impl Fp128Field {
    pub fn new() -> Self {
        Self
    }

    pub fn zero() -> Fp128Element {
        Fp128Element::zero()
    }

    pub fn one() -> Fp128Element {
        Fp128Element::one()
    }

    pub fn from_u64(val: u64) -> Fp128Element {
        Fp128Element::from_raw(val)
    }

    pub fn from_i64(val: i64) -> Fp128Element {
        Fp128Element::from_i64(val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_basic() {
        let a = Fp128Element::from_raw(100);
        let b = Fp128Element::from_raw(200);
        assert_eq!((a + b).0, 300);
    }

    #[test]
    fn test_add_wrap() {
        let max = Fp128Element::from_raw(Fp128Element::MODULUS - 1);
        let one = Fp128Element::one();
        let result = max + one;
        assert_eq!(result.0, 0); // MODULUS ≡ 0
    }

    #[test]
    fn test_sub_basic() {
        let a = Fp128Element::from_raw(100);
        let b = Fp128Element::from_raw(200);
        let diff = a - b;
        assert_eq!(diff.0, Fp128Element::MODULUS - 100);
    }

    #[test]
    fn test_sub_underflow() {
        let zero = Fp128Element::zero();
        let one = Fp128Element::one();
        let diff = zero - one;
        assert_eq!(diff.0, Fp128Element::MODULUS - 1);
    }

    #[test]
    fn test_mul_basic() {
        let two = Fp128Element::from_raw(2);
        let three = Fp128Element::from_raw(3);
        assert_eq!((two * three).0, 6);
    }

    #[test]
    fn test_mul_overflow() {
        // Test multiplication that overflows u64
        let a = Fp128Element::from_raw(Fp128Element::MODULUS - 1);
        let two = Fp128Element::from_raw(2);
        let result = a * two;
        // (q-1) * 2 = 2q - 2 ≡ -2 ≡ q - 2 (mod q)
        assert_eq!(result.0, Fp128Element::MODULUS - 2);
    }

    #[test]
    fn test_neg() {
        let a = Fp128Element::from_raw(42);
        let neg = -a;
        assert_eq!(neg.0, Fp128Element::MODULUS - 42);

        let zero = Fp128Element::zero();
        assert_eq!(-zero, zero);
    }

    #[test]
    fn test_inv() {
        let two = Fp128Element::from_raw(2);
        let inv_two = two.inv().unwrap();

        // 2 * (1/2) = 1 (mod q)
        let result = two * inv_two;
        assert!(result.is_one());
    }

    #[test]
    fn test_inv_zero() {
        let zero = Fp128Element::zero();
        assert!(zero.inv().is_none());
    }

    #[test]
    fn test_pow() {
        let base = Fp128Element::from_raw(2);
        let exp = 10;
        let result = base.pow(exp);

        // 2^10 = 1024
        assert_eq!(result.0, 1024);
    }

    #[test]
    fn test_pow_identity() {
        let a = Fp128Element::from_raw(42);

        // a^0 = 1
        assert!(a.pow(0).is_one());

        // a^1 = a
        assert_eq!(a.pow(1).0, 42);
    }

    #[test]
    fn test_fermat_little_theorem() {
        // a^(q-1) = 1 for a != 0
        let a = Fp128Element::from_raw(42);
        let exp = Fp128Element::MODULUS - 1;
        let result = a.pow(exp);

        assert!(result.is_one());
    }

    #[test]
    fn test_primitive_root_of_unity() {
        // Test for n = 4 (2^2)
        let omega = Fp128Element::primitive_root_of_unity(2);

        // omega^4 should equal 1
        let omega_4 = omega.pow(4);
        assert!(omega_4.is_one());

        // omega^2 should equal -1 (i.e., q - 1)
        let omega_2 = omega.pow(2);
        assert_eq!(omega_2.0, Fp128Element::MODULUS - 1);

        // omega^1 should NOT equal 1
        assert!(!omega.is_one());
    }

    #[test]
    fn test_primitive_root_of_unity_powers() {
        for log_n in 1..8 {
            let n = 1u64 << log_n;
            let omega = Fp128Element::primitive_root_of_unity(log_n);

            // omega^n = 1
            assert!(omega.pow(n).is_one(), "omega^{} != 1 for log_n={}", n, log_n);

            // omega^(n/2) = -1
            let omega_half = omega.pow(n / 2);
            assert_eq!(
                omega_half.0,
                Fp128Element::MODULUS - 1,
                "omega^(n/2) != -1 for log_n={}",
                log_n
            );
        }
    }

    #[test]
    fn test_field_axioms() {
        // Addition associativity: (a + b) + c = a + (b + c)
        let a = Fp128Element::from_raw(100);
        let b = Fp128Element::from_raw(200);
        let c = Fp128Element::from_raw(300);

        assert_eq!((a + b) + c, a + (b + c));

        // Addition commutativity: a + b = b + a
        assert_eq!(a + b, b + a);

        // Multiplication associativity: (a * b) * c = a * (b * c)
        assert_eq!((a * b) * c, a * (b * c));

        // Multiplication commutativity: a * b = b * a
        assert_eq!(a * b, b * a);

        // Distributivity: a * (b + c) = a*b + a*c
        assert_eq!(a * (b + c), a * b + a * c);

        // Identity elements
        assert_eq!(a + Fp128Element::zero(), a);
        assert_eq!(a * Fp128Element::one(), a);
    }

    #[test]
    fn test_multiplicative_inverse() {
        // Test inverses for various values
        let test_values = [1u64, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];

        for &val in &test_values {
            let a = Fp128Element::from_raw(val);
            let inv_a = a.inv().expect("Should have inverse");

            // a * a^(-1) = 1
            let result = a * inv_a;
            assert!(
                result.is_one(),
                "Inverse failed for {}: got {}",
                val,
                result.0
            );
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        use crate::primitives::serialization::{Compress, DoryDeserialize, DorySerialize, Validate};

        let original = Fp128Element::from_raw(0xDEADBEEFCAFEBABE);

        // Serialize
        let mut buffer = Vec::new();
        original
            .serialize_with_mode(&mut buffer, Compress::Yes)
            .unwrap();

        // Deserialize
        let recovered =
            Fp128Element::deserialize_with_mode(buffer.as_slice(), Compress::Yes, Validate::Yes)
                .unwrap();

        assert_eq!(original.0, recovered.0);
    }

    #[test]
    fn test_from_i64() {
        // Positive values
        assert_eq!(Fp128Element::from_i64(42).0, 42);
        assert_eq!(Fp128Element::from_i64(0).0, 0);

        // Negative values
        let neg_one = Fp128Element::from_i64(-1);
        assert_eq!(neg_one.0, Fp128Element::MODULUS - 1);

        let neg_42 = Fp128Element::from_i64(-42);
        assert_eq!(neg_42.0, Fp128Element::MODULUS - 42);
    }
}
