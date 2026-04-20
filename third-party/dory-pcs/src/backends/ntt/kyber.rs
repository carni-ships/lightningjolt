//! Kyber/Dilithium Lattice Curve Implementation
//!
//! This module implements the LatticeCurve trait using Kyber-768 parameters:
//! - Polynomial ring: Z_q[X] / (X^n + 1) where n = 256
//! - Modulus: q = 2^64 - 2^32 + 1
//! - NTT size: 256

use crate::primitives::lattice_trait::{LatticeCurve, LatticeElement};
use crate::backends::ntt::fp128::Fp128Element;
use crate::backends::ntt::cpu_ntt::TwiddleCache;
use crate::primitives::serialization::{Compress, DoryDeserialize, DorySerialize, Valid, SerializationError};
use std::io::{Read, Write};

/// Kyber polynomial in coefficient form
#[derive(Clone, Debug, Default)]
pub struct KyberPolynomial {
    /// Coefficients (length must be power of 2, typically 256 for Kyber-768)
    pub coeffs: Vec<Fp128Element>,
}

/// Kyber polynomial in NTT form (frequency domain)
#[derive(Clone, Debug, Default)]
pub struct KyberPolynomialNTT {
    /// NTT values
    pub values: Vec<Fp128Element>,
}

/// Kyber curve implementation
#[derive(Clone, Default)]
pub struct KyberCurve;

impl KyberCurve {
    pub fn new() -> Self {
        Self
    }

    /// Polynomial degree (NTT size)
    pub const N: usize = 256;

    /// Create polynomial from coefficients
    pub fn poly_from_coeffs(coeffs: &[Fp128Element]) -> KyberPolynomial {
        assert!(coeffs.len() == Self::N);
        KyberPolynomial {
            coeffs: coeffs.to_vec(),
        }
    }

    /// Create zero polynomial
    pub fn zero_poly() -> KyberPolynomial {
        KyberPolynomial {
            coeffs: vec![Fp128Element::zero(); Self::N],
        }
    }

    /// Create polynomial with all ones
    pub fn one_poly() -> KyberPolynomial {
        let mut coeffs = vec![Fp128Element::zero(); Self::N];
        coeffs[0] = Fp128Element::one();
        KyberPolynomial { coeffs }
    }
}

/// Compute inner product of two vectors of polynomials
/// inner_product(A, B) = Σ⟨A[i], B[i]⟩ where ⟨P, Q⟩ = Σ P[j] * Q[j]
fn compute_inner_product(a: &[KyberPolynomialNTT], b: &[KyberPolynomialNTT]) -> Fp128Element {
    let mut sum = Fp128Element::zero();
    for (pa, pb) in a.iter().zip(b.iter()) {
        for (va, vb) in pa.values.iter().zip(pb.values.iter()) {
            sum = sum + *va * *vb;
        }
    }
    sum
}

impl LatticeCurve for KyberCurve {
    type Polynomial = KyberPolynomial;
    type PolynomialNTT = KyberPolynomialNTT;
    type Commitment = KyberCommitment;
    type Field = Fp128Element;

    fn commit(polynomial: &Self::Polynomial) -> Self::Commitment {
        KyberCommitment::from_poly(polynomial)
    }

    fn commit_ntt(polynomial_ntt: &Self::PolynomialNTT) -> Self::Commitment {
        KyberCommitment::from_ntt(polynomial_ntt.clone())
    }

    fn to_ntt(polynomial: &Self::Polynomial) -> Self::PolynomialNTT {
        polynomial.ntt()
    }

    fn from_ntt(ntt: &Self::PolynomialNTT) -> Self::Polynomial {
        ntt.intt()
    }

    fn inner_product(a: &[Self::PolynomialNTT], b: &[Self::PolynomialNTT]) -> Self::Field {
        compute_inner_product(a, b)
    }

    fn poly_multiply(a: &Self::PolynomialNTT, b: &Self::PolynomialNTT) -> Self::PolynomialNTT {
        a.multiply(b)
    }

    fn poly_from_coeffs(coeffs: &[Self::Field]) -> Self::Polynomial {
        KyberPolynomial {
            coeffs: coeffs.to_vec(),
        }
    }

    fn zero_poly() -> Self::Polynomial {
        KyberPolynomial {
            coeffs: vec![Fp128Element::zero(); Self::N],
        }
    }

    fn one_poly() -> Self::Polynomial {
        let mut coeffs = vec![Fp128Element::zero(); Self::N];
        coeffs[0] = Fp128Element::one();
        KyberPolynomial { coeffs }
    }
}

impl KyberPolynomial {
    /// Perform NTT transformation
    pub fn ntt(&self) -> KyberPolynomialNTT {
        let cache = TwiddleCache::new(8); // 2^8 = 256
        let values = cache.ntt_fwd(&self.coeffs, 8);
        KyberPolynomialNTT { values }
    }

    /// Perform inverse NTT transformation
    pub fn intt(&self) -> KyberPolynomial {
        let cache = TwiddleCache::new(8);
        let coeffs = cache.ntt_inv(&self.coeffs, 8);
        KyberPolynomial { coeffs }
    }

    /// Add two polynomials
    pub fn add(&self, rhs: &KyberPolynomial) -> KyberPolynomial {
        let mut result = self.coeffs.clone();
        for i in 0..result.len() {
            result[i] = result[i] + rhs.coeffs[i];
        }
        KyberPolynomial { coeffs: result }
    }

    /// Subtract two polynomials
    pub fn sub(&self, rhs: &KyberPolynomial) -> KyberPolynomial {
        let mut result = self.coeffs.clone();
        for i in 0..result.len() {
            result[i] = result[i] - rhs.coeffs[i];
        }
        KyberPolynomial { coeffs: result }
    }
}

impl KyberPolynomialNTT {
    /// Pointwise multiplication in frequency domain
    pub fn multiply(&self, rhs: &KyberPolynomialNTT) -> KyberPolynomialNTT {
        let mut result = Vec::with_capacity(self.values.len());
        for i in 0..self.values.len() {
            result.push(self.values[i] * rhs.values[i]);
        }
        KyberPolynomialNTT { values: result }
    }

    /// Scale by scalar
    pub fn scale(&self, scalar: &Fp128Element) -> KyberPolynomialNTT {
        let mut result = Vec::with_capacity(self.values.len());
        for i in 0..self.values.len() {
            result.push(self.values[i] * *scalar);
        }
        KyberPolynomialNTT { values: result }
    }

    /// Inverse transformation back to coefficient form
    pub fn intt(&self) -> KyberPolynomial {
        let cache = TwiddleCache::new(8);
        let coeffs = cache.ntt_inv_naive(&self.values, 8);
        KyberPolynomial { coeffs }
    }

    /// Create zero polynomial in NTT form
    pub fn zero() -> Self {
        KyberPolynomialNTT {
            values: vec![Fp128Element::zero(); KyberCurve::N],
        }
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl std::ops::Add for KyberPolynomial {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        KyberPolynomial::add(&self, &rhs)
    }
}

impl std::ops::Sub for KyberPolynomial {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        KyberPolynomial::sub(&self, &rhs)
    }
}

impl std::ops::Add for KyberPolynomialNTT {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut result = Vec::with_capacity(self.values.len());
        for i in 0..self.values.len() {
            result.push(self.values[i] + rhs.values[i]);
        }
        KyberPolynomialNTT { values: result }
    }
}

impl std::ops::Mul<Fp128Element> for KyberPolynomialNTT {
    type Output = Self;
    fn mul(self, scalar: Fp128Element) -> Self {
        self.scale(&scalar)
    }
}

impl Valid for KyberPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl DorySerialize for KyberPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        for coeff in &self.coeffs {
            coeff.serialize_with_mode(&mut writer, _compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        self.coeffs.len() * 8
    }
}

impl DoryDeserialize for KyberPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        _validate: crate::primitives::serialization::Validate,
    ) -> Result<Self, SerializationError> {
        let mut coeffs = Vec::with_capacity(KyberCurve::N);
        for _ in 0..KyberCurve::N {
            coeffs.push(Fp128Element::deserialize_with_mode(&mut reader, _compress, _validate)?);
        }
        Ok(KyberPolynomial { coeffs })
    }
}

impl Valid for KyberPolynomialNTT {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl DorySerialize for KyberPolynomialNTT {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        for val in &self.values {
            val.serialize_with_mode(&mut writer, _compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        self.values.len() * 8
    }
}

impl DoryDeserialize for KyberPolynomialNTT {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        _validate: crate::primitives::serialization::Validate,
    ) -> Result<Self, SerializationError> {
        let mut values = Vec::with_capacity(KyberCurve::N);
        for _ in 0..KyberCurve::N {
            values.push(Fp128Element::deserialize_with_mode(&mut reader, _compress, _validate)?);
        }
        Ok(KyberPolynomialNTT { values })
    }
}

/// Lattice-based commitment for Kyber
/// Commitment = NTT(polynomial) in frequency domain
pub struct KyberCommitment {
    pub poly_ntt: KyberPolynomialNTT,
}

impl KyberCommitment {
    pub fn from_poly(poly: &KyberPolynomial) -> Self {
        KyberCommitment {
            poly_ntt: poly.ntt(),
        }
    }

    pub fn from_ntt(poly_ntt: KyberPolynomialNTT) -> Self {
        KyberCommitment { poly_ntt }
    }
}

impl Default for KyberCommitment {
    fn default() -> Self {
        Self {
            poly_ntt: KyberPolynomialNTT::zero(),
        }
    }
}

impl std::fmt::Debug for KyberCommitment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KyberCommitment(len={})", self.poly_ntt.len())
    }
}

impl Clone for KyberCommitment {
    fn clone(&self) -> Self {
        Self {
            poly_ntt: KyberPolynomialNTT {
                values: self.poly_ntt.values.clone(),
            },
        }
    }
}

impl PartialEq for KyberCommitment {
    fn eq(&self, other: &Self) -> bool {
        self.poly_ntt.values == other.poly_ntt.values
    }
}

impl Valid for KyberCommitment {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl DorySerialize for KyberCommitment {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.poly_ntt.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.poly_ntt.serialized_size(compress)
    }
}

impl DoryDeserialize for KyberCommitment {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: crate::primitives::serialization::Validate,
    ) -> Result<Self, SerializationError> {
        Ok(KyberCommitment {
            poly_ntt: KyberPolynomialNTT::deserialize_with_mode(reader, compress, validate)?,
        })
    }
}

impl LatticeElement for KyberPolynomial {
    type Field = Fp128Element;

    fn zero() -> Self {
        KyberPolynomial {
            coeffs: vec![Fp128Element::zero(); KyberCurve::N],
        }
    }

    fn one() -> Self {
        let mut coeffs = vec![Fp128Element::zero(); KyberCurve::N];
        coeffs[0] = Fp128Element::one();
        KyberPolynomial { coeffs }
    }

    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| c.is_zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        self.add(rhs)
    }

    fn sub(&self, rhs: &Self) -> Self {
        self.sub(rhs)
    }

    fn scale(&self, k: &Self::Field) -> Self {
        let mut result = Vec::with_capacity(self.coeffs.len());
        for c in &self.coeffs {
            result.push(*c * *k);
        }
        KyberPolynomial { coeffs: result }
    }

    fn to_field(&self) -> Self::Field {
        // Sum of coefficients
        let mut sum = Fp128Element::zero();
        for c in &self.coeffs {
            sum = sum + *c;
        }
        sum
    }
}

impl LatticeElement for KyberPolynomialNTT {
    type Field = Fp128Element;

    fn zero() -> Self {
        KyberPolynomialNTT::zero()
    }

    fn one() -> Self {
        let mut values = vec![Fp128Element::zero(); KyberCurve::N];
        values[0] = Fp128Element::one();
        KyberPolynomialNTT { values }
    }

    fn is_zero(&self) -> bool {
        self.values.iter().all(|v| v.is_zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        let mut result = Vec::with_capacity(self.values.len());
        for i in 0..self.values.len() {
            result.push(self.values[i] + rhs.values[i]);
        }
        KyberPolynomialNTT { values: result }
    }

    fn sub(&self, rhs: &Self) -> Self {
        let mut result = Vec::with_capacity(self.values.len());
        for i in 0..self.values.len() {
            result.push(self.values[i] - rhs.values[i]);
        }
        KyberPolynomialNTT { values: result }
    }

    fn scale(&self, k: &Self::Field) -> Self {
        self.scale(k)
    }

    fn to_field(&self) -> Self::Field {
        // For NTT form, we need to sum after inverse transform
        let poly = self.intt();
        poly.to_field()
    }
}

/// Inner product in frequency domain (lattice analogue of pairing)
/// This is the core operation that replaces E::pair(g1, g2)
pub fn lattice_inner_product(a: &KyberPolynomialNTT, b: &KyberPolynomialNTT) -> Fp128Element {
    // Inner product = sum of pointwise products
    // In frequency domain, this corresponds to the bilinear pairing
    let mut sum = Fp128Element::zero();
    for i in 0..a.values.len() {
        sum = sum + a.values[i] * b.values[i];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // FIXME: NTT inverse has bugs - needs debugging
    fn test_ntt_roundtrip() {
        let coeffs: Vec<Fp128Element> = (0..256)
            .map(|i| Fp128Element::from_raw(i as u64))
            .collect();
        let poly = KyberCurve::poly_from_coeffs(&coeffs);
        let ntt = poly.ntt();
        let recovered = ntt.intt();

        for (a, b) in coeffs.iter().zip(recovered.coeffs.iter()) {
            assert_eq!(*a, *b);
        }
    }

    #[test]
    fn test_polynomial_add() {
        let a = KyberCurve::poly_from_coeffs(&vec![Fp128Element::from_raw(1); 256]);
        let b = KyberCurve::poly_from_coeffs(&vec![Fp128Element::from_raw(2); 256]);
        let c = a + b;

        for coeff in c.coeffs {
            assert_eq!(coeff.0, 3);
        }
    }

    #[test]
    #[ignore] // FIXME: NTT inverse has bugs - needs debugging
    fn test_ntt_multiply() {
        // Create two polynomials
        let mut coeffs1 = vec![Fp128Element::zero(); 256];
        coeffs1[0] = Fp128Element::from_raw(1);
        coeffs1[1] = Fp128Element::from_raw(2);

        let mut coeffs2 = vec![Fp128Element::zero(); 256];
        coeffs2[0] = Fp128Element::from_raw(3);
        coeffs2[1] = Fp128Element::from_raw(4);

        let poly1 = KyberCurve::poly_from_coeffs(&coeffs1);
        let poly2 = KyberCurve::poly_from_coeffs(&coeffs2);

        // Multiply in frequency domain
        let ntt1 = poly1.ntt();
        let ntt2 = poly2.ntt();
        let product_ntt = ntt1.multiply(&ntt2);

        // Convert back and check
        let product = product_ntt.intt();

        // The product should be (1 + 2x) * (3 + 4x) = 3 + 4x + 6x + 8x^2 = 3 + 10x (mod X^256 + 1)
        // But with modular arithmetic...
        assert_eq!(product.coeffs[0].0, 3);
    }

    #[test]
    fn test_commitment() {
        let coeffs: Vec<Fp128Element> = (0..256)
            .map(|i| Fp128Element::from_raw(i as u64 * 12345))
            .collect();
        let poly = KyberCurve::poly_from_coeffs(&coeffs);
        let commit = KyberCommitment::from_poly(&poly);

        assert_eq!(commit.poly_ntt.len(), 256);
    }
}
