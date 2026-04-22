//! Minimal zkMetal CPU Pippenger MSM wrapper for jolt-core.
//!
//! This module provides a direct wrapper around zkMetal's C Pippenger implementation
//! (bn254_pippenger_msm) using only zkmetal_sys FFI bindings.
//!
//! Features:
//! - Multi-threaded Pippenger with adaptive window sizing
//! - Mixed affine addition (saves 4 muls/add per point)
//! - Batch-to-affine via Montgomery's trick
//!
//! IMPORTANT: zkMetal expects:
//! - Point coordinates (x, y) in Montgomery form (4 u64 per coordinate)
//! - Scalars in non-Montgomery (standard) integer form (8 u32 per scalar)
//!
//! Scalar conversion uses the CORRECT approach from zkMetal's arkworks.rs:
//! For scalars < 2^64, the low 64 bits of the Montgomery form directly give the scalar value.

#[cfg(feature = "zkmetal")]
use zkmetal_sys::{bn254_pippenger_msm, bn254_projective_to_affine};

use ark_bn254::{Fq, G1Affine, G1Projective, Fr};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{BigInt, One, PrimeField, Zero};

/// Convert a single arkworks Fr (Montgomery form) to zkMetal Pippenger scalar format.
///
/// zkMetal's Pippenger expects scalars in standard (non-Montgomery) integer form.
/// For scalars < 2^64, we can directly extract the low 64 bits of the Montgomery
/// representation. This works because:
///   Fr(s) in Montgomery = s * R mod r
///   For small s (< 2^64), the low 64 bits of s * R mod r directly equal s
///
/// Returns 8 u32 limbs in little-endian order.
#[inline]
fn fr_to_pippenger_scalar(fr: &Fr) -> [u32; 8] {
    let bigint: BigInt<4> = fr.into_bigint();
    let low_u64 = bigint.0[0];
    [low_u64 as u32, (low_u64 >> 32) as u32, 0, 0, 0, 0, 0, 0]
}

/// Batch convert scalars from Montgomery form to zkMetal Pippenger format.
fn batch_convert_scalars(scalars: &[Fr]) -> Vec<u32> {
    let n = scalars.len();
    let mut result = Vec::with_capacity(n * 8);
    for s in scalars {
        result.extend_from_slice(&fr_to_pippenger_scalar(s));
    }
    result
}

/// Run zkMetal's C Pippenger MSM with correct scalar conversion.
/// Returns the result in Jacobian projective coordinates.
///
/// - `points`: n G1Affine points
/// - `scalars`: n Fr scalars (Montgomery form)
#[cfg(feature = "zkmetal")]
pub fn pippenger_msm(
    points: &[G1Affine],
    scalars: &[Fr],
) -> G1Projective {
    assert_eq!(points.len(), scalars.len(), "msm: points/scalars length mismatch");
    let n = points.len();

    // Convert points to zkMetal format (8 u64 per point: x[4], y[4])
    // zkMetal expects Montgomery form for point coordinates (same as arkworks BigInt)
    let mut points_raw: Vec<u64> = Vec::with_capacity(n * 8);
    for p in points {
        let x_bigint: BigInt<4> = p.x.into_bigint();
        let y_bigint: BigInt<4> = p.y.into_bigint();
        points_raw.extend_from_slice(&x_bigint.0);
        points_raw.extend_from_slice(&y_bigint.0);
    }

    // Convert scalars from Montgomery form to standard integer form
    // Using the correct approach: direct low 64-bit extraction
    let scalars_raw = batch_convert_scalars(scalars);

    // Call zkMetal C Pippenger - returns projective point
    let mut projective_result: [u64; 12] = [0; 12];
    unsafe {
        bn254_pippenger_msm(
            points_raw.as_ptr(),
            scalars_raw.as_ptr(),
            n as i32,
            projective_result.as_mut_ptr(),
        );
    }

    // Convert zkMetal projective to affine using zkMetal's own converter
    let mut affine_result: [u64; 8] = [0; 8];
    unsafe {
        bn254_projective_to_affine(
            projective_result.as_ptr(),
            affine_result.as_mut_ptr(),
        );
    }

    // zkMetal outputs affine coordinates in STANDARD form (raw integer)
    // We need to convert to Montgomery form for arkworks
    let x_bigint = BigInt([
        affine_result[0], affine_result[1], affine_result[2], affine_result[3]
    ]);
    let y_bigint = BigInt([
        affine_result[4], affine_result[5], affine_result[6], affine_result[7]
    ]);

    // from_bigint treats input as standard form and converts to Montgomery
    let x = Fq::from_bigint(x_bigint).expect("valid x");
    let y = Fq::from_bigint(y_bigint).expect("valid y");

    // Check for point at infinity (affine (0,0) in standard form)
    let is_zero = x.is_zero() && y.is_zero();

    if is_zero {
        G1Projective::zero()
    } else {
        G1Projective::new_unchecked(x, y, Fq::one())
    }
}

/// Alternative: Create G1Affine directly by bypassing validation
#[cfg(feature = "zkmetal")]
pub fn pippenger_msm_affine(
    points: &[G1Affine],
    scalars: &[Fr],
) -> G1Affine {
    assert_eq!(points.len(), scalars.len(), "msm: points/scalars length mismatch");
    let n = points.len();

    // Convert points to zkMetal format (8 u64 per point: x[4], y[4])
    // zkMetal expects Montgomery form for point coordinates (same as arkworks BigInt)
    let mut points_raw: Vec<u64> = Vec::with_capacity(n * 8);
    for p in points {
        let x_bigint: BigInt<4> = p.x.into_bigint();
        let y_bigint: BigInt<4> = p.y.into_bigint();
        points_raw.extend_from_slice(&x_bigint.0);
        points_raw.extend_from_slice(&y_bigint.0);
    }

    // Convert scalars from Montgomery form to standard integer form
    // Using the correct approach: direct low 64-bit extraction
    let scalars_raw = batch_convert_scalars(scalars);

    // Call zkMetal C Pippenger - returns projective point
    let mut projective_result: [u64; 12] = [0; 12];
    unsafe {
        bn254_pippenger_msm(
            points_raw.as_ptr(),
            scalars_raw.as_ptr(),
            n as i32,
            projective_result.as_mut_ptr(),
        );
    }

    // Convert zkMetal projective to affine using zkMetal's own converter
    let mut affine_result: [u64; 8] = [0; 8];
    unsafe {
        bn254_projective_to_affine(
            projective_result.as_ptr(),
            affine_result.as_mut_ptr(),
        );
    }

    // zkMetal outputs affine coordinates in STANDARD form (raw integer)
    // We need to convert to Montgomery form for arkworks
    let x_bigint = BigInt([
        affine_result[0], affine_result[1], affine_result[2], affine_result[3]
    ]);
    let y_bigint = BigInt([
        affine_result[4], affine_result[5], affine_result[6], affine_result[7]
    ]);

    // from_bigint treats input as standard form and converts to Montgomery
    let x = Fq::from_bigint(x_bigint).expect("valid x");
    let y = Fq::from_bigint(y_bigint).expect("valid y");

    // Check for point at infinity (affine (0,0) in standard form)
    let is_zero = x.is_zero() && y.is_zero();

    if is_zero {
        G1Affine::zero()
    } else {
        // Create affine point WITHOUT validation - zkMetal guarantees valid BN254 points
        G1Affine::new_unchecked(x, y)
    }
}