//! Lattice Curve Trait for NTT-Based Polynomial Commitments
//!
//! This module defines the LatticeCurve trait that provides the algebraic
//! operations needed for Dory-style protocols using lattice/NTT operations
//! instead of elliptic curve pairings.
//!
//! Key mappings from EC to Lattice:
//! - `E::pair(g1, g2)` → `lattice_inner_product(ntt(g1), ntt(g2))`
//! - `M1::msm(bases, scalars)` → `ntt_multiply(base_ntt, scalars_ntt)`
//! - Cross-term `⟨Qi, Qj⟩` → Polynomial multiplication in frequency domain

use crate::primitives::serialization::{DoryDeserialize, DorySerialize};

/// Lattice element - polynomial in coefficient or NTT form
pub trait LatticeElement: Clone + Send + Sync + DorySerialize + DoryDeserialize {
    /// The field this element lives in
    type Field: LatticeField;

    /// Create zero element
    fn zero() -> Self;

    /// Create identity element (1 for multiplicative, 0 for additive)
    fn one() -> Self;

    /// Check if zero
    fn is_zero(&self) -> bool;

    /// Add two elements
    fn add(&self, rhs: &Self) -> Self;

    /// Subtract two elements
    fn sub(&self, rhs: &Self) -> Self;

    /// Scale by field element
    fn scale(&self, k: &Self::Field) -> Self;

    /// Get underlying field value (for base elements)
    fn to_field(&self) -> Self::Field;
}

/// Field trait for lattice operations
pub trait LatticeField:
    Sized
    + Clone
    + Copy
    + PartialEq
    + Send
    + Sync
    + DorySerialize
    + DoryDeserialize
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
    + for<'a> std::ops::Add<&'a Self, Output = Self>
    + for<'a> std::ops::Sub<&'a Self, Output = Self>
    + for<'a> std::ops::Mul<&'a Self, Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;

    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
    fn inv(self) -> Option<Self>;

    fn random() -> Self;
    fn from_u64(val: u64) -> Self;
    fn from_i64(val: i64) -> Self;
}

/// Lattice Curve trait providing polynomial commitment operations
/// Analogous to PairingCurve but using NTT/lattice operations
pub trait LatticeCurve: Clone {
    /// Polynomial ring element (in coefficient form)
    type Polynomial: LatticeElement<Field = Self::Field>;

    /// Polynomial in frequency domain (NTT form)
    type PolynomialNTT: LatticeElement<Field = Self::Field>;

    /// Commitment type - polynomial in NTT form
    type Commitment: Clone + Default + DorySerialize + DoryDeserialize + PartialEq;

    /// The field elements live in
    type Field: LatticeField;

    /// Commit a polynomial to the frequency domain
    fn commit(polynomial: &Self::Polynomial) -> Self::Commitment;

    /// Commit from NTT form directly
    fn commit_ntt(polynomial_ntt: &Self::PolynomialNTT) -> Self::Commitment;

    /// Convert polynomial to NTT form
    fn to_ntt(polynomial: &Self::Polynomial) -> Self::PolynomialNTT;

    /// Convert from NTT form back to coefficient form
    fn from_ntt(ntt: &Self::PolynomialNTT) -> Self::Polynomial;

    /// Inner product in frequency domain (replaces pairing)
    /// For vectors of polynomials A, B: inner_product(A, B) = Σ⟨A[i], B[i]⟩
    /// where ⟨A[i], B[i]⟩ = Σ A[i][j] * B[i][j] (sum of pointwise products)
    fn inner_product(a: &[Self::PolynomialNTT], b: &[Self::PolynomialNTT]) -> Self::Field;

    /// Polynomial multiplication in frequency domain
    fn poly_multiply(a: &Self::PolynomialNTT, b: &Self::PolynomialNTT) -> Self::PolynomialNTT;

    /// Create polynomial from coefficients
    fn poly_from_coeffs(coeffs: &[Self::Field]) -> Self::Polynomial;

    /// Create zero polynomial
    fn zero_poly() -> Self::Polynomial;

    /// Create polynomial representing 1
    fn one_poly() -> Self::Polynomial;
}
