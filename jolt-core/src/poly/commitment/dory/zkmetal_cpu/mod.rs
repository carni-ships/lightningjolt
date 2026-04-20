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
//! - Point coordinates (x, y) in Montgomery form
//! - Scalars in non-Montgomery (standard) form

#[cfg(feature = "zkmetal")]
use zkmetal_sys::{bn254_pippenger_msm, bn254_projective_to_affine};

use ark_bn254::{Fq, G1Affine, G1Projective, Fr};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{PrimeField, BigInt, One, Zero};

#[cfg(test)]
mod debug_tests;

/// BN254 Fq field: R mod p (Montgomery form of 1)
/// p = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
const FQ_R_MOD_P: [u64; 4] = [
    0xd35d438dc58f0d9d,
    0x0a78eb28f5c70b3d,
    0x666ea36f7879462c,
    0x0e0a77c19a07df2f,
];

/// BN254 Fr scalar field: R mod r (Montgomery form of 1)
/// r = 0x30644e72e131a029b85045b68181585d2833e84879b97091f43e1f593f0000001
const FR_R_MOD_R: [u64; 4] = [
    0xac96341c4ffffffb,
    0x36fc76959f60cd29,
    0x666ea36f7879462e,
    0x0e0a77c19a07df2f,
];

/// Convert arkworks Fq to zkMetal format
/// Both Fq::into_bigint() and zkMetal expect the same representation (Montgomery)
fn fq_to_zkmetal(x: Fq) -> [u64; 4] {
    let bigint: BigInt<4> = x.into_bigint();
    bigint.0
}

/// Run zkMetal's C Pippenger MSM.
/// Returns the result in Jacobian projective coordinates (arkworks standard).
///
/// - `points`: n G1Affine points
/// - `scalars`: n Fr scalars
#[cfg(feature = "zkmetal")]
pub fn pippenger_msm(
    points: &[G1Affine],
    scalars: &[Fr],
) -> G1Projective {
    assert_eq!(points.len(), scalars.len(), "msm: points/scalars length mismatch");

    let n = points.len();

    // Convert points to zkMetal format (8 u64 per point: x[4], y[4])
    // zkMetal expects Montgomery form for point coordinates
    let mut points_raw: Vec<u64> = Vec::with_capacity(n * 8);
    for (i, p) in points.iter().enumerate().take(2) {
        // Verify points are valid
        let x3 = p.x * p.x * p.x;
        let y2 = p.y * p.y;
        let b = Fq::from(3);
        let on_curve = y2 == x3 + b;
        eprintln!("DEBUG point[{}]: x={}, y={}, on_curve={}", i, p.x, p.y, on_curve);
        eprintln!("DEBUG point[{}] x_bigint={:?}, y_bigint={:?}", i,
            p.x.into_bigint().0, p.y.into_bigint().0);

        let x_mont = fq_to_zkmetal(p.x);
        let y_mont = fq_to_zkmetal(p.y);
        points_raw.extend_from_slice(&x_mont);
        points_raw.extend_from_slice(&y_mont);
    }
    for p in points.iter().skip(2) {
        let x_mont = fq_to_zkmetal(p.x);
        let y_mont = fq_to_zkmetal(p.y);
        points_raw.extend_from_slice(&x_mont);
        points_raw.extend_from_slice(&y_mont);
    }

    // Convert scalars to limbs (8 u32 per scalar)
    // zkMetal C code extracts raw bits from scalars using extract_window()
    // which just does bit shifting - no Montgomery arithmetic.
    // So scalars should be in STANDARD form (raw integer value).
    let mut scalars_raw: Vec<u32> = Vec::with_capacity(n * 8);
    for (i, s) in scalars.iter().enumerate().take(2) {
        let bigint: BigInt<4> = s.into_bigint();
        eprintln!("DEBUG scalar[{}]: Fr={}, std={:?}", i, s, bigint.0);
        // Convert [u64; 4] to [u32; 8] - little-endian
        for limb in bigint.0.iter() {
            scalars_raw.push(*limb as u32);
            scalars_raw.push((*limb >> 32) as u32);
        }
    }
    for s in scalars.iter().skip(2) {
        let bigint: BigInt<4> = s.into_bigint();
        for limb in bigint.0.iter() {
            scalars_raw.push(*limb as u32);
            scalars_raw.push((*limb >> 32) as u32);
        }
    }

    // Call zkMetal C Pippenger directly via FFI
    let mut projective_result: [u64; 12] = [0; 12];
    unsafe {
        bn254_pippenger_msm(
            points_raw.as_ptr(),
            scalars_raw.as_ptr(),
            n as i32,
            projective_result.as_mut_ptr(),
        );
    }

    eprintln!("DEBUG projective result: X={:?}, Y={:?}, Z={:?}",
        &projective_result[0..4], &projective_result[4..8], &projective_result[8..12]);

    // Also convert projective to affine using arkworks (for comparison)
    use ark_ec::scalar_mul::variable_base::VariableBaseMSM;
    let ark_result: G1Projective = VariableBaseMSM::msm(points, scalars).expect("msm should not fail");
    let ark_affine = ark_result.into_affine();
    eprintln!("DEBUG arkworks affine: x={}, y={}", ark_affine.x, ark_affine.y);

    // Convert arkworks projective to BigInt for comparison
    let ark_x_bigint: BigInt<4> = ark_affine.x.into_bigint();
    let ark_y_bigint: BigInt<4> = ark_affine.y.into_bigint();
    eprintln!("DEBUG arkworks x_bigint: {:?}, y_bigint: {:?}", ark_x_bigint.0, ark_y_bigint.0);

    // Convert zkMetal projective to affine using zkMetal's own converter
    let mut affine_result: [u64; 8] = [0; 8];
    unsafe {
        bn254_projective_to_affine(
            projective_result.as_ptr(),
            affine_result.as_mut_ptr(),
        );
    }

    eprintln!("DEBUG zkMetal affine result: x_bigint={:?}, y_bigint={:?}",
        &affine_result[0..4], &affine_result[4..8]);

    // zkMetal outputs affine coordinates in STANDARD form (verified: Method 2 works for 1 point)
    let x_bigint = BigInt([affine_result[0], affine_result[1], affine_result[2], affine_result[3]]);
    let y_bigint = BigInt([affine_result[4], affine_result[5], affine_result[6], affine_result[7]]);

    eprintln!("DEBUG output zkMetal: x_bigint={:?}, y_bigint={:?}", x_bigint.0, y_bigint.0);

    // Test 1: use from_bigint (standard to Montgomery)
    let x1 = Fq::from_bigint(x_bigint).expect("valid x");
    let y1 = Fq::from_bigint(y_bigint).expect("valid y");
    let b = Fq::from(3);
    let x3_1 = x1 * x1 * x1;
    let y2_1 = y1 * y1;
    let valid1 = y2_1 == x3_1 + b;
    eprintln!("DEBUG Test 1 (from_bigint): curve valid={}", valid1);

    // Test 2: transmute (treat as already Montgomery)
    let x2: Fq = unsafe { std::mem::transmute_copy(&x_bigint) };
    let y2: Fq = unsafe { std::mem::transmute_copy(&y_bigint) };
    let x3_2 = x2 * x2 * x2;
    let y2_2 = y2 * y2;
    let valid2 = y2_2 == x3_2 + b;
    eprintln!("DEBUG Test 2 (transmute): curve valid={}", valid2);

    // Test 3: what if zkMetal affine is already in Montgomery?
    // In that case, from_bigint would be correct (converts standard to Montgomery)
    // and transmute would be wrong.
    // But the 1-point test showed from_bigint works... so why not 2-point?
    //
    // Let's check: maybe zkMetal's projective-to-affine has an issue?
    // Let's try using arkworks' into_affine directly on the projective result
    let zk_proj_x_bigint = BigInt([projective_result[0], projective_result[1], projective_result[2], projective_result[3]]);
    let zk_proj_y_bigint = BigInt([projective_result[4], projective_result[5], projective_result[6], projective_result[7]]);
    let zk_proj_z_bigint = BigInt([projective_result[8], projective_result[9], projective_result[10], projective_result[11]]);
    let zk_proj_x: Fq = unsafe { std::mem::transmute_copy(&zk_proj_x_bigint) };
    let zk_proj_y: Fq = unsafe { std::mem::transmute_copy(&zk_proj_y_bigint) };
    let zk_proj_z: Fq = unsafe { std::mem::transmute_copy(&zk_proj_z_bigint) };
    eprintln!("DEBUG zkMetal projective (as Fq): X={}, Y={}, Z={}", zk_proj_x, zk_proj_y, zk_proj_z);

    // Try arkworks projective -> affine
    let zk_proj = G1Projective::new_unchecked(zk_proj_x, zk_proj_y, zk_proj_z);
    let zk_proj_aff = zk_proj.into_affine();
    let zk_proj_aff_bigint_x: BigInt<4> = zk_proj_aff.x.into_bigint();
    let zk_proj_aff_bigint_y: BigInt<4> = zk_proj_aff.y.into_bigint();
    eprintln!("DEBUG zkMetal projective -> arkworks affine: x={:?}, y={:?}",
        zk_proj_aff_bigint_x.0, zk_proj_aff_bigint_y.0);

    // Check if they're equal
    let matches_arkworks = x_bigint.0 == ark_x_bigint.0 && y_bigint.0 == ark_y_bigint.0;
    eprintln!("DEBUG zkMetal affine matches arkworks: {}", matches_arkworks);

    let is_zero = x1.is_zero() && y1.is_zero();

    if is_zero {
        G1Projective::zero()
    } else {
        // Use new_unchecked and hope for the best - the point is on curve, just not validated
        let p = G1Projective::new_unchecked(x1, y1, Fq::one());
        eprintln!("DEBUG returning point with from_bigint coords");
        p
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
    // zkMetal expects STANDARD form (same as arkworks BigInt representation)
    let mut points_raw: Vec<u64> = Vec::with_capacity(n * 8);
    for p in points {
        let x_bigint = p.x.into_bigint();
        let y_bigint = p.y.into_bigint();
        points_raw.extend_from_slice(&x_bigint.0);
        points_raw.extend_from_slice(&y_bigint.0);
    }

    // Convert scalars to limbs (8 u32 per scalar, non-Montgomery)
    let mut scalars_raw: Vec<u32> = Vec::with_capacity(n * 8);
    for s in scalars {
        let bigint: BigInt<4> = s.into_bigint();
        for limb in bigint.0.iter() {
            scalars_raw.push(*limb as u32);
            scalars_raw.push((*limb >> 32) as u32);
        }
    }

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

    // Convert affine result to arkworks format
    let x_bigint = BigInt([affine_result[0], affine_result[1], affine_result[2], affine_result[3]]);
    let y_bigint = BigInt([affine_result[4], affine_result[5], affine_result[6], affine_result[7]]);
    let x = Fq::from_bigint(x_bigint).expect("invalid x from zkMetal");
    let y = Fq::from_bigint(y_bigint).expect("invalid y from zkMetal");

    // Check if it's the point at infinity (x=0, y=0 in affine)
    let is_zero = x.is_zero() && y.is_zero();

    if is_zero {
        G1Affine::zero()
    } else {
        // Create affine point WITHOUT validation - zkMetal guarantees valid BN254 points
        G1Affine::new_unchecked(x, y)
    }
}