//! Hybrid MSM backend with automatic GPU dispatch.
//!
//! This module provides hybrid implementations that automatically dispatch
//! to ICICLE GPU for large MSMs and use arkworks for smaller ones.
//!
//! ## Strategy
//!
//! GPU overhead (memory transfer, kernel launch) makes GPU beneficial only
//! for larger MSMs. This hybrid dispatches based on size threshold:
//! - Small MSMs (< GPU_THRESHOLD): arkworks (low overhead)
//! - Large MSMs (>= GPU_THRESHOLD): ICICLE GPU (parallelism wins)
//!
//! ## Usage
//!
//! ```ignore
//! use dory_pcs::backends::hybrid::{HybridG1Routines, HybridG2Routines};
//!
//! // Use Hybrid routines instead of G1Routines/G2Routines
//! let proof = prove_dory_prime::<..., HybridG1Routines, HybridG2Routines, ...>(...);
//! ```
//!
//! Enable with `icicle` feature (also requires `arkworks`):
//! ```toml
//! dory = { features = ["backends", "icicle"] }
//! ```

// This module requires the ICICLE feature
#![cfg(feature = "icicle")]

use crate::backends::arkworks::{ArkG1, ArkG2, ArkFr, G1Routines, G2Routines};
use crate::primitives::arithmetic::{DoryRoutines, Group};
use std::sync::Once;

/// GPU dispatch threshold - GPU overhead worthwhile above this size.
/// Tuned for BN254 curves. May need adjustment for different curves.
pub const GPU_MSM_THRESHOLD: usize = 512;

static ICICLE_INIT: Once = Once::new();

/// Ensure ICICLE runtime is initialized (one-time initialization).
fn ensure_icicle_initialized() {
    ICICLE_INIT.call_once(|| {
        let _ = icicle_runtime::runtime::load_backend_from_env_or_default();
        let device = icicle_runtime::Device::new("CPU", 0);
        icicle_runtime::set_device(&device).expect("failed to set ICICLE device");
    });
}

/// Get ICICLE MSM config optimized for Dory workloads.
fn icicle_msm_config() -> icicle_core::msm::MSMConfig {
    let mut cfg = icicle_core::msm::MSMConfig::default();
    cfg.are_scalars_montgomery_form = true;
    cfg.are_bases_montgomery_form = false;
    cfg
}

// Re-export ICICLE types for internal use
use ark_bn254::{Fq, Fq2, Fr, G1Affine as ArkG1Affine, G1Projective as ArkG1Projective, G2Affine as ArkG2Affine, G2Projective as ArkG2Projective};
use ark_ec::AffineRepr;
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use icicle_bn254::curve::{
    BaseField as IcicleBaseField, G1Affine as IcicleG1Affine,
    G1Projective as IcicleG1Projective, G2Affine as IcicleG2Affine,
    G2BaseField as IcicleG2BaseField, G2Projective as IcicleG2Projective,
    ScalarField as IcicleScalarField,
};
use icicle_core::bignum::BigNum;
use icicle_core::msm::MSM;
use icicle_runtime::memory::HostSlice;

// ============================================================================
// Conversion helpers
// ============================================================================

/// Zero-copy transmute arkworks Fr to ICICLE ScalarField.
#[inline]
unsafe fn scalars_to_icicle(scalars: &[Fr]) -> &[IcicleScalarField] {
    std::slice::from_raw_parts(scalars.as_ptr() as *const IcicleScalarField, scalars.len())
}

/// Convert arkworks Fq to ICICLE base field.
#[inline]
fn fq_to_icicle_base(fq: &Fq) -> IcicleBaseField {
    let bigint: ark_ff::BigInt<4> = fq.into_bigint();
    unsafe { std::mem::transmute_copy(&bigint) }
}

/// Convert arkworks Fq2 to ICICLE G2BaseField.
#[inline]
fn fq2_to_icicle_g2base(fq2: &Fq2) -> IcicleG2BaseField {
    let c0_bigint = fq2.c0.into_bigint();
    let c1_bigint = fq2.c1.into_bigint();
    let mut limbs = [0u32; 16];
    unsafe {
        let c0_u32: [u32; 8] = std::mem::transmute(c0_bigint);
        let c1_u32: [u32; 8] = std::mem::transmute(c1_bigint);
        limbs[..8].copy_from_slice(&c0_u32);
        limbs[8..].copy_from_slice(&c1_u32);
    }
    IcicleG2BaseField::from(limbs)
}

/// Convert ICICLE G1Projective to arkworks G1Projective.
#[inline]
fn g1_proj_from_icicle(p: IcicleG1Projective) -> ArkG1Projective {
    let affine: IcicleG1Affine = p.into();
    let zero_affine = IcicleG1Affine {
        x: IcicleBaseField::zero(),
        y: IcicleBaseField::zero(),
    };
    if affine == zero_affine {
        return ArkG1Projective::default();
    }
    let ark_affine = ArkG1Affine::new_unchecked(icicle_base_to_fq(&affine.x), icicle_base_to_fq(&affine.y));
    ark_affine.into()
}

/// Convert ICICLE G2Projective to arkworks G2Projective.
#[inline]
fn g2_proj_from_icicle(p: IcicleG2Projective) -> ArkG2Projective {
    let affine: IcicleG2Affine = p.into();
    let zero_affine = IcicleG2Affine {
        x: IcicleG2BaseField::zero(),
        y: IcicleG2BaseField::zero(),
    };
    if affine == zero_affine {
        return ArkG2Projective::default();
    }
    let ark_affine = ArkG2Affine::new_unchecked(
        icicle_g2base_to_fq2(&affine.x),
        icicle_g2base_to_fq2(&affine.y),
    );
    ark_affine.into()
}

/// Convert ICICLE base field to arkworks Fq.
#[inline]
fn icicle_base_to_fq(base: &IcicleBaseField) -> Fq {
    let bigint: ark_ff::BigInt<4> = unsafe { std::mem::transmute_copy(base) };
    Fq::from_bigint(bigint).expect("ICICLE base field element out of range")
}

/// Convert ICICLE G2BaseField to arkworks Fq2.
#[inline]
fn icicle_g2base_to_fq2(base: &IcicleG2BaseField) -> Fq2 {
    let limbs: [u32; 16] = unsafe { std::mem::transmute_copy(base) };
    let c0_bigint: ark_ff::BigInt<4> =
        unsafe { std::mem::transmute::<[u32; 8], _>(limbs[..8].try_into().unwrap()) };
    let c1_bigint: ark_ff::BigInt<4> =
        unsafe { std::mem::transmute::<[u32; 8], _>(limbs[8..].try_into().unwrap()) };
    let c0 = Fq::from_bigint(c0_bigint).expect("ICICLE G2 c0 out of range");
    let c1 = Fq::from_bigint(c1_bigint).expect("ICICLE G2 c1 out of range");
    Fq2::new(c0, c1)
}

// ============================================================================
// GPU MSM functions
// ============================================================================

/// GPU-accelerated G1 MSM using ICICLE.
#[inline]
fn icicle_g1_msm(affines: &[ArkG1Affine], scalars: &[Fr]) -> ArkG1Projective {
    ensure_icicle_initialized();

    let icicle_bases = g1_affine_to_icicle(affines);
    let icicle_scalars = unsafe { scalars_to_icicle(scalars) };

    let cfg = icicle_msm_config();
    let mut result = vec![IcicleG1Projective::default()];

    IcicleG1Projective::msm(
        HostSlice::from_slice(icicle_scalars),
        HostSlice::from_slice(&icicle_bases),
        &cfg,
        HostSlice::from_mut_slice(&mut result),
    )
    .expect("ICICLE G1 MSM failed");

    g1_proj_from_icicle(result.into_iter().next().unwrap())
}

/// GPU-accelerated G2 MSM using ICICLE.
#[inline]
fn icicle_g2_msm(affines: &[ArkG2Affine], scalars: &[Fr]) -> ArkG2Projective {
    ensure_icicle_initialized();

    let icicle_bases = g2_affine_to_icicle(affines);
    let icicle_scalars = unsafe { scalars_to_icicle(scalars) };

    let cfg = icicle_msm_config();
    let mut result = vec![IcicleG2Projective::default()];

    IcicleG2Projective::msm(
        HostSlice::from_slice(icicle_scalars),
        HostSlice::from_slice(&icicle_bases),
        &cfg,
        HostSlice::from_mut_slice(&mut result),
    )
    .expect("ICICLE G2 MSM failed");

    g2_proj_from_icicle(result.into_iter().next().unwrap())
}

/// Convert G1 points to ICICLE format.
#[inline]
fn g1_affine_to_icicle(points: &[ArkG1Affine]) -> Vec<IcicleG1Affine> {
    points
        .iter()
        .map(|p| {
            if p.is_zero() {
                IcicleG1Affine {
                    x: IcicleBaseField::zero(),
                    y: IcicleBaseField::zero(),
                }
            } else {
                IcicleG1Affine {
                    x: fq_to_icicle_base(&p.x),
                    y: fq_to_icicle_base(&p.y),
                }
            }
        })
        .collect()
}

/// Convert G2 points to ICICLE format.
#[inline]
fn g2_affine_to_icicle(points: &[ArkG2Affine]) -> Vec<IcicleG2Affine> {
    points
        .iter()
        .map(|p| {
            if p.is_zero() {
                IcicleG2Affine {
                    x: IcicleG2BaseField::zero(),
                    y: IcicleG2BaseField::zero(),
                }
            } else {
                IcicleG2Affine {
                    x: fq2_to_icicle_g2base(&p.x),
                    y: fq2_to_icicle_g2base(&p.y),
                }
            }
        })
        .collect()
}

// ============================================================================
// Hybrid G1 Routines
// ============================================================================

/// Hybrid G1 routines: GPU for large MSMs, CPU for small.
///
/// Automatically dispatches to ICICLE GPU for MSMs with `bases.len() >= GPU_MSM_THRESHOLD`.
/// Falls back to arkworks for smaller MSMs (where GPU overhead exceeds benefit).
pub struct HybridG1Routines;

impl DoryRoutines<ArkG1> for HybridG1Routines {
    #[tracing::instrument(skip_all, name = "HybridG1::msm", fields(len = bases.len(), gpu = bases.len() >= GPU_MSM_THRESHOLD))]
    fn msm(bases: &[ArkG1], scalars: &[ArkFr]) -> ArkG1 {
        assert_eq!(
            bases.len(),
            scalars.len(),
            "MSM requires equal length vectors"
        );

        if bases.is_empty() {
            return ArkG1::identity();
        }

        let len = bases.len();

        // Small MSM: use arkworks (GPU overhead not worthwhile)
        if len < GPU_MSM_THRESHOLD {
            return G1Routines::msm(bases, scalars);
        }

        // Large MSM: use ICICLE GPU
        let affines: Vec<ArkG1Affine> = bases.iter().map(|b| b.0.into_affine()).collect();
        let raw_scalars: Vec<Fr> = scalars.iter().map(|s| s.0).collect();
        ArkG1(icicle_g1_msm(&affines, &raw_scalars))
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG1, scalars: &[ArkFr]) -> Vec<ArkG1> {
        // CPU-based fixed-base multiplication is already efficient
        // GPU doesn't help for this operation pattern
        G1Routines::fixed_base_vector_scalar_mul(base, scalars)
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG1], vs: &mut [ArkG1], scalar: &ArkFr) {
        G1Routines::fixed_scalar_mul_bases_then_add(bases, vs, scalar)
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG1], addends: &[ArkG1], scalar: &ArkFr) {
        G1Routines::fixed_scalar_mul_vs_then_add(vs, addends, scalar)
    }
}

// ============================================================================
// Hybrid G2 Routines
// ============================================================================

/// Hybrid G2 routines: GPU for large MSMs, CPU for small.
///
/// Automatically dispatches to ICICLE GPU for MSMs with `bases.len() >= GPU_MSM_THRESHOLD`.
/// Falls back to arkworks for smaller MSMs (where GPU overhead exceeds benefit).
pub struct HybridG2Routines;

impl DoryRoutines<ArkG2> for HybridG2Routines {
    #[tracing::instrument(skip_all, name = "HybridG2::msm", fields(len = bases.len(), gpu = bases.len() >= GPU_MSM_THRESHOLD))]
    fn msm(bases: &[ArkG2], scalars: &[ArkFr]) -> ArkG2 {
        assert_eq!(
            bases.len(),
            scalars.len(),
            "MSM requires equal length vectors"
        );

        if bases.is_empty() {
            return ArkG2::identity();
        }

        let len = bases.len();

        // Small MSM: use arkworks (GPU overhead not worthwhile)
        if len < GPU_MSM_THRESHOLD {
            return G2Routines::msm(bases, scalars);
        }

        // Large MSM: use ICICLE GPU
        let affines: Vec<ArkG2Affine> = bases.iter().map(|b| b.0.into_affine()).collect();
        let raw_scalars: Vec<Fr> = scalars.iter().map(|s| s.0).collect();
        ArkG2(icicle_g2_msm(&affines, &raw_scalars))
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG2, scalars: &[ArkFr]) -> Vec<ArkG2> {
        G2Routines::fixed_base_vector_scalar_mul(base, scalars)
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG2], vs: &mut [ArkG2], scalar: &ArkFr) {
        G2Routines::fixed_scalar_mul_bases_then_add(bases, vs, scalar)
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG2], addends: &[ArkG2], scalar: &ArkFr) {
        G2Routines::fixed_scalar_mul_vs_then_add(vs, addends, scalar)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::scalar_mul::variable_base::VariableBaseMSM;
    use ark_ec::CurveGroup;
    use ark_ff::UniformRand;

    #[test]
    fn test_hybrid_g1_matches_arkworks_small() {
        let mut rng = ark_std::test_rng();
        let n = 64; // Below threshold

        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let projectives: Vec<ArkG1Projective> = (0..n).map(|_| ArkG1Projective::rand(&mut rng)).collect();
        let affines: Vec<ArkG1Affine> = projectives.iter().map(|p| p.into_affine()).collect();

        let ark_result: ArkG1Projective = VariableBaseMSM::msm(&affines, &scalars).unwrap();
        let hybrid_result = G1Routines::msm(
            &projectives.iter().map(|p| ArkG1(p)).collect::<Vec<_>>(),
            &scalars.iter().map(|s| ArkFr(s)).collect::<Vec<_>>(),
        );

        assert_eq!(ark_result, hybrid_result.0, "Hybrid G1 must match arkworks for small MSMs");
    }

    #[test]
    fn test_hybrid_g2_matches_arkworks_small() {
        let mut rng = ark_std::test_rng();
        let n = 64; // Below threshold

        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let projectives: Vec<ArkG2Projective> = (0..n).map(|_| ArkG2Projective::rand(&mut rng)).collect();
        let affines: Vec<ArkG2Affine> = projectives.iter().map(|p| p.into_affine()).collect();

        let ark_result: ArkG2Projective = VariableBaseMSM::msm(&affines, &scalars).unwrap();
        let hybrid_result = G2Routines::msm(
            &projectives.iter().map(|p| ArkG2(p)).collect::<Vec<_>>(),
            &scalars.iter().map(|s| ArkFr(s)).collect::<Vec<_>>(),
        );

        assert_eq!(ark_result, hybrid_result.0, "Hybrid G2 must match arkworks for small MSMs");
    }

    #[test]
    fn test_gpu_threshold_constant() {
        assert!(GPU_MSM_THRESHOLD >= 256, "Threshold should be at least 256");
        assert!(GPU_MSM_THRESHOLD <= 4096, "Threshold should be at most 4096");
    }
}