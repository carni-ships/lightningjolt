#![allow(missing_docs)]

use super::{DoryDeserialize, DorySerialize};

pub trait Field:
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

    /// Returns the least significant bit of this field element.
    /// Used for balanced branching in tree-parallel protocols.
    ///
    /// This extracts the LSB from the field element's canonical byte representation,
    /// giving a uniform 50/50 split over the field.
    fn branch_bit(&self) -> bool;
}

pub trait Group:
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
    + std::ops::Neg<Output = Self>
    + for<'a> std::ops::Add<&'a Self, Output = Self>
    + for<'a> std::ops::Sub<&'a Self, Output = Self>
{
    type Scalar: Field
        + std::ops::Mul<Self, Output = Self>
        + for<'a> std::ops::Mul<&'a Self, Output = Self>;

    fn identity() -> Self;
    fn add(&self, rhs: &Self) -> Self;
    fn neg(&self) -> Self;
    fn scale(&self, k: &Self::Scalar) -> Self;

    fn random() -> Self;
}

pub trait PairingCurve: Clone {
    type G1: Group;
    type G2: Group;
    type GT: Group; // Multiplicative subgroup F^* of the extension field

    /// e : G1 × G2 → GT
    fn pair(p: &Self::G1, q: &Self::G2) -> Self::GT;

    /// Π e(p_i, q_i)
    fn multi_pair(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair requires equal length vectors"
        );

        if ps.is_empty() {
            return Self::GT::identity();
        }

        ps.iter()
            .zip(qs.iter())
            .fold(Self::GT::identity(), |acc, (p, q)| {
                acc.add(&Self::pair(p, q))
            })
    }

    /// Optimized multi-pairing when G2 points come from setup/generators
    ///
    /// This variant should be used when the G2 points are from the prover setup
    /// (e.g., g2_vec generators). Backend implementations can optimize this by
    /// caching prepared G2 points.
    ///
    /// # Parameters
    /// - `ps`: G1 points (typically computed values like row commitments or v-vectors)
    /// - `qs`: G2 points from setup (e.g., `setup.g2_vec[..n]`)
    ///
    /// # Returns
    /// Product of pairings: Π e(p_i, q_i)
    ///
    /// # Default Implementation
    /// Delegates to `multi_pair`
    fn multi_pair_g2_setup(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        Self::multi_pair(ps, qs)
    }

    /// Optimized multi-pairing when G1 points are from the prover setup.
    ///
    /// This variant should be used when the G1 points are from the prover setup
    /// (e.g., g1_vec generators). Backend implementations can optimize this by
    /// caching prepared G1 points.
    ///
    /// # Parameters
    /// - `ps`: G1 points from setup (e.g., `setup.g1_vec[..n]`)
    /// - `qs`: G2 points (typically computed values like v-vectors)
    ///
    /// # Returns
    /// Product of pairings: Π e(p_i, q_i)
    ///
    /// # Default Implementation
    /// Delegates to `multi_pair`
    fn multi_pair_g1_setup(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        Self::multi_pair(ps, qs)
    }

    /// Batch pairing that returns individual GT results for each pair.
    ///
    /// Unlike `multi_pair` which returns the product of all pairings, this returns
    /// a vector of individual pairing results, enabling efficient batching when each
    /// result needs to be processed separately (e.g., with different masks).
    ///
    /// # Parameters
    /// - `ps`: G1 points
    /// - `qs`: G2 points
    ///
    /// # Returns
    /// Vector of GT elements: [e(p_0, q_0), e(p_1, q_1), ..., e(p_n-1, q_n-1)]
    ///
    /// # Default Implementation
    /// Calls `multi_pair` for each pair individually (for backends without batch support)
    fn multi_pair_batch(ps: &[Self::G1], qs: &[Self::G2]) -> Vec<Self::GT> {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair_batch requires equal length vectors"
        );
        ps.iter()
            .zip(qs.iter())
            .map(|(p, q)| Self::pair(p, q))
            .collect()
    }

    /// Batch pairing with G2 from setup, returning individual GT results.
    ///
    /// # Default Implementation
    /// Delegates to `multi_pair_batch`
    fn multi_pair_g2_setup_batch(ps: &[Self::G1], qs: &[Self::G2]) -> Vec<Self::GT> {
        Self::multi_pair_batch(ps, qs)
    }

    /// Batch pairing with G1 from setup, returning individual GT results.
    ///
    /// # Default Implementation
    /// Delegates to `multi_pair_batch`
    fn multi_pair_g1_setup_batch(ps: &[Self::G1], qs: &[Self::G2]) -> Vec<Self::GT> {
        Self::multi_pair_batch(ps, qs)
    }

    /// Compute two separate pairing products using a single Miller loop.
    ///
    /// Given pairs (a0[i], b0[i]) and (a1[i], b1[i]) for i=0..n,
    /// computes Π e(a0[i], b0[i]) and Π e(a1[i], b1[i]) with one Miller loop.
    ///
    /// Returns (product0, product1) where:
    /// - product0 = Π e(a0[i], b0[i])  (first set of pairs)
    /// - product1 = Π e(a1[i], b1[i])  (second set of pairs)
    fn multi_pair_two_products(
        a0: &[Self::G1],
        b0: &[Self::G2],
        a1: &[Self::G1],
        b1: &[Self::G2],
    ) -> (Self::GT, Self::GT);
}

/// Dory requires MSMs and vector scaling ops, hence we expose a trait for optimized versions of such routines.
pub trait DoryRoutines<G: Group> {
    fn msm(bases: &[G], scalars: &[G::Scalar]) -> G;

    /// Fixed-base vectorized scalar multiplication where the same base is scaled by each scalar individually
    /// Computes: \[base * scalars\[0\], base * scalars\[1\], ..., base * scalars\[n-1\]\]
    fn fixed_base_vector_scalar_mul(base: &G, scalars: &[G::Scalar]) -> Vec<G>;

    /// vs\[i\] = vs\[i\] + scalar * bases\[i\]
    fn fixed_scalar_mul_bases_then_add(bases: &[G], vs: &mut [G], scalar: &G::Scalar);

    /// vs\[i\] = scalar * vs\[i\] + addends\[i\]
    fn fixed_scalar_mul_vs_then_add(vs: &mut [G], addends: &[G], scalar: &G::Scalar);

    /// Fold field vectors: left\[i\] = left\[i\] * scalar + right\[i\]
    fn fold_field_vectors(left: &mut [G::Scalar], right: &[G::Scalar], scalar: &G::Scalar) {
        assert_eq!(left.len(), right.len(), "Lengths must match");
        for i in 0..left.len() {
            left[i] = left[i] * *scalar + right[i];
        }
    }

    /// Fold 4 field vectors into one using 4-ary coefficients.
    ///
    /// Computes: output[i] = c0*q0[i] + c1*q1[i] + c2*q2[i] + c3*q3[i]
    /// where coeffs = (c0, c1, c2, c3) with sum(c_i) = 1.
    fn fold_4ary_field_vectors(
        output: &mut [G::Scalar],
        quarters: [&[G::Scalar]; 4],
        coeffs: &(G::Scalar, G::Scalar, G::Scalar, G::Scalar),
    ) {
        assert_eq!(output.len(), quarters[0].len(), "Output length must match quarter length");
        for i in 0..output.len() {
            output[i] = coeffs.0 * quarters[0][i]
                + coeffs.1 * quarters[1][i]
                + coeffs.2 * quarters[2][i]
                + coeffs.3 * quarters[3][i];
        }
    }

    /// Fold 4 group element vectors into one using 4-ary coefficients.
    ///
    /// Computes: output[i] = c0*q0[i] + c1*q1[i] + c2*q2[i] + c3*q3[i]
    /// where coeffs = (c0, c1, c2, c3) with sum(c_i) = 1.
    fn fold_4ary_group_vectors(
        output: &mut [G],
        quarters: [&[G]; 4],
        coeffs: &(G::Scalar, G::Scalar, G::Scalar, G::Scalar),
    ) {
        assert_eq!(output.len(), quarters[0].len(), "Output length must match quarter length");
        for i in 0..output.len() {
            output[i] = quarters[0][i].scale(&coeffs.0)
                + quarters[1][i].scale(&coeffs.1)
                + quarters[2][i].scale(&coeffs.2)
                + quarters[3][i].scale(&coeffs.3);
        }
    }
}
