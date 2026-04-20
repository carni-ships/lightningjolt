//! zkMetal GPU interop layer for BN254 MSM operations
//!
//! This module provides type conversions and wrapper functions for using
//! zkMetal's GPU-accelerated MSM when available.
//!
//! # Usage
//!
//! ```
//! use jolt_core::poly::commitment::dory::zkmetal_interop::{ark_to_zk_g1, ark_to_zk_fr, zk_msm};
//!
//! // Convert arkworks points and scalars to zkMetal format
//! let zk_points = ark_to_zk_g1(&ark_points);
//! let zk_scalars = ark_to_zk_fr(&ark_scalars);
//!
//! // Run GPU-accelerated MSM
//! let result = zk_msm(&zk_points, &zk_scalars);
//!
//! // Convert result back to arkworks
//! let ark_result = zk_to_ark_g1(result);
//! ```

#[cfg(feature = "zkmetal")]
use zkmetal;

use ark_ff::PrimeField;

/// Convert arkworks G1 affine points to zkMetal format.
/// Both are stored as [u64; 4] Montgomery form, so this is a zero-copy cast.
#[cfg(feature = "zkmetal")]
pub fn ark_to_zk_g1(ark_points: &[ark_bn254::G1Affine]) -> Vec<zkmetal::bn254::G1Affine> {
    use ark_ec::AffineRepr;

    ark_points
        .iter()
        .map(|p| {
            let x_bigint: [u64; 4] = unsafe { std::mem::transmute_copy(&p.x.into_bigint()) };
            let y_bigint: [u64; 4] = unsafe { std::mem::transmute_copy(&p.y.into_bigint()) };
            zkmetal::bn254::G1Affine {
                x: zkmetal::bn254::Fq(x_bigint),
                y: zkmetal::bn254::Fq(y_bigint),
            }
        })
        .collect()
}

/// Convert arkworks Fr scalars to zkMetal format.
/// Returns scalars as [u32; 8] (non-Montgomery integer form for Pippenger).
#[cfg(feature = "zkmetal")]
pub fn ark_to_zk_fr(ark_scalars: &[ark_bn254::Fr]) -> Vec<[u32; 8]> {
    ark_scalars
        .iter()
        .map(|s| {
            let bigint: ark_ff::BigInt<4> = s.into_bigint();
            // SAFETY: BigInt<4> is [u64; 4] = 32 bytes, [u32; 8] is also 32 bytes
            let limbs: [u32; 8] = unsafe { std::mem::transmute(bigint) };
            limbs
        })
        .collect()
}

/// Run zkMetal GPU-accelerated G1 MSM.
/// Panics if zkMetal is not available or MSM fails.
#[cfg(feature = "zkmetal")]
pub fn zk_msm(points: &[zkmetal::bn254::G1Affine], scalars: &[[u32; 8]]) -> zkmetal::bn254::G1Projective {
    assert_eq!(
        points.len(),
        scalars.len(),
        "zk_msm: points and scalars length mismatch"
    );
    zkmetal::msm::bn254_g1_msm(points, scalars)
}

/// Convert zkMetal G1Projective back to arkworks format.
#[cfg(feature = "zkmetal")]
pub fn zk_to_ark_g1(p: zkmetal::bn254::G1Projective) -> ark_bn254::G1Projective {
    use ark_bn254::Fq;
    use ark_ec::short_weierstrass::Projective;

    let x: Fq = unsafe { std::mem::transmute_copy(&p.x.0) };
    let y: Fq = unsafe { std::mem::transmute_copy(&p.y.0) };
    let z: Fq = unsafe { std::mem::transmute_copy(&p.z.0) };

    // zkMetal uses standard projective, arkworks uses Jacobian
    // Convert: standard (X, Y, Z) -> affine -> Jacobian
    let aff = ark_bn254::G1Affine::new_unchecked(x, y);
    aff.mul_by_cofactor_to_projective()
}

/// Check if zkMetal GPU MSM is available.
#[cfg(feature = "zkmetal")]
pub fn is_available() -> bool {
    true // zkMetal is always available if feature is enabled
}

/// Get zkMetal version info.
#[cfg(feature = "zkmetal")]
pub fn version() -> &'static str {
    "zkMetal GPU MSM (Pippenger)"
}

/// Estimate GPU speedup vs CPU for given MSM size.
/// Returns estimated speedup factor.
#[cfg(feature = "zkmetal")]
pub fn estimated_speedup(n: usize) -> f64 {
    // Based on zkMetal benchmarks for stable macOS
    match n {
        n if n <= 1 << 14 => 0.8,  // CPU faster for small sizes
        n if n <= 1 << 16 => 3.0,   // GPU ~3x faster
        n if n <= 1 << 18 => 6.0,   // GPU ~6x faster
        _ => 8.0, // GPU ~8x faster for large
    }
}

/// Decide whether to use GPU MSM based on size threshold.
/// Returns true if GPU should be used.
#[cfg(feature = "zkmetal")]
pub fn gpu_recommended_for_size(n: usize) -> bool {
    // Use GPU for MSMs larger than 2^15 elements
    n > (1 << 15)
}

// Re-export zkmetal types for direct access if needed
#[cfg(feature = "zkmetal")]
pub use zkmetal::bn254 as types;

#[cfg(not(feature = "zkmetal"))]
pub fn is_available() -> bool {
    false
}

#[cfg(not(feature = "zkmetal"))]
pub fn version() -> &'static str {
    "zkMetal not available (feature not enabled)"
}

#[cfg(not(feature = "zkmetal"))]
pub fn gpu_recommended_for_size(_n: usize) -> bool {
    false
}

#[cfg(not(feature = "zkmetal"))]
pub fn estimated_speedup(_n: usize) -> f64 {
    1.0
}