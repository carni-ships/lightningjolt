//! Batch proof generation utilities for Dory
//!
//! This module provides utilities for combining multiple Dory proofs into a single
//! batch proof using the homomorphic properties of Dory commitments.
//!
//! ## Efficiency
//!
//! - **Commitment combination**: O(num_polys) GT/G1 operations
//! - **Proof generation**: O(1) extra work beyond single proof
//! - **Verification**: O(1) extra pairings per additional polynomial

//! See tests for usage examples.

use crate::primitives::arithmetic::{Group, PairingCurve};

/// Combines multiple tier-2 (GT) commitments into a single commitment using RLC
///
/// Given commitments Com(P_i) = tier_2_i, computes:
/// Com(∑ r_i * P_i) = ∑ r_i * Com(P_i) = ∑ r_i * tier_2_i
///
/// # Parameters
/// - `commitments`: Slice of tier-2 commitments (GT elements)
/// - `coeffs`: Random coefficients for the linear combination
///
/// # Returns
/// Combined tier-2 commitment
pub fn combine_commitments<'a, E: PairingCurve>(
    commitments: &[E::GT],
    coeffs: &[<E::GT as Group>::Scalar],
) -> E::GT
where
    E::GT: Clone,
{
    assert_eq!(commitments.len(), coeffs.len());

    if commitments.is_empty() {
        return E::GT::identity();
    }

    let mut combined = commitments[0].clone().scale(&coeffs[0]);
    for i in 1..commitments.len() {
        let scaled = commitments[i].clone().scale(&coeffs[i]);
        combined = combined + scaled;
    }
    combined
}

/// Combines multiple sets of row commitments into a single set using RLC
///
/// Given row commitments row_j(P_i) for each polynomial, computes:
/// row_j(∑ r_i * P_i) = ∑ r_i * row_j(P_i)
///
/// # Parameters
/// - `row_commitments`: Vec of Vec of G1 elements, each inner vec is one polynomial's row commitments
/// - `coeffs`: Random coefficients for the linear combination
///
/// # Returns
/// Combined row commitments (Vec<G1>)
pub fn combine_row_commitments<'a, E: PairingCurve>(
    row_commitments: &[Vec<E::G1>],
    coeffs: &[<E::G1 as Group>::Scalar],
) -> Vec<E::G1>
where
    E::G1: Clone,
{
    assert_eq!(row_commitments.len(), coeffs.len());

    if row_commitments.is_empty() {
        return vec![];
    }

    let num_rows = row_commitments[0].len();
    let mut combined = vec![E::G1::identity(); num_rows];

    for (poly_idx, poly_rows) in row_commitments.iter().enumerate() {
        assert_eq!(poly_rows.len(), num_rows);
        for (row_idx, row_commitment) in poly_rows.iter().enumerate() {
            let scaled = row_commitment.clone().scale(&coeffs[poly_idx]);
            combined[row_idx] = combined[row_idx].clone() + scaled;
        }
    }

    combined
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::arkworks::{ArkFr, ArkG1, BN254};
    use crate::primitives::arithmetic::Field;

    #[test]
    fn test_combine_commitments() {
        let commitment1 = <BN254 as PairingCurve>::GT::random();
        let commitment2 = <BN254 as PairingCurve>::GT::random();
        let commitment3 = <BN254 as PairingCurve>::GT::random();

        let commitments = vec![commitment1.clone(), commitment2.clone(), commitment3.clone()];
        let coeffs = vec![ArkFr::from_u64(1), ArkFr::from_u64(2), ArkFr::from_u64(3)];

        let combined = combine_commitments::<BN254>(&commitments, &coeffs);

        // Verify: 1*c1 + 2*c2 + 3*c3
        let expected = commitment1 + (commitment2 + commitment2) + (commitment3 + commitment3 + commitment3);
        assert_eq!(combined, expected);
    }

    #[test]
    fn test_combine_row_commitments() {
        // 2 polynomials, 4 rows each
        let poly1_rows = vec![
            ArkG1::random(),
            ArkG1::random(),
            ArkG1::random(),
            ArkG1::random(),
        ];
        let poly2_rows = vec![
            ArkG1::random(),
            ArkG1::random(),
            ArkG1::random(),
            ArkG1::random(),
        ];

        let row_commitments = vec![poly1_rows.clone(), poly2_rows.clone()];
        let coeffs = vec![ArkFr::from_u64(2), ArkFr::from_u64(3)];

        let combined = combine_row_commitments::<BN254>(&row_commitments, &coeffs);

        assert_eq!(combined.len(), 4);
        // row_i = 2*poly1_row_i + 3*poly2_row_i
        for i in 0..4 {
            let expected = poly1_rows[i] + poly1_rows[i] + poly2_rows[i] + poly2_rows[i] + poly2_rows[i];
            assert_eq!(combined[i], expected);
        }
    }
}
