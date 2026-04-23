//! Custom DoryRoutines implementations using jolt_optimizations and ICICLE GPU
//!
//! ## GPU-Adaptive MSM Dispatch
//!
//! ICICLE GPU acceleration is used only when the MSM size is >= `GPU_MSM_THRESHOLD`.
//! For smaller MSMs, arkworks CPU is faster due to lower overhead.
//!
//! Threshold = 512 elements was chosen to balance:
//! - GPU kernel launch overhead (~0.1-0.5ms)
//! - MSM compute time (scales with size)
//! - Memory transfer overhead
//!
//! For Dory-Prime with nu=sigma:
//! - 2^6=64: no GPU (too small)
//! - 2^9=512: GPU kicks in for VMV and R&F phases
//! - 2^12=4096: full GPU acceleration across all phases

use super::wrappers::{ArkFr, ArkG1, ArkG2};
use ark_bn254::{Fr, G1Projective, G2Projective};
use ark_ec::scalar_mul::variable_base::VariableBaseMSM as ArkVariableBaseMSM;
use ark_ec::CurveGroup;
use dory::primitives::arithmetic::DoryRoutines;
use rayon::prelude::*;

/// GPU dispatch threshold for Dory-Prime MSMs.
///
/// zkMetal NEON is DISABLED due to scalar format incompatibility with arkworks.
/// All MSMs use arkworks until zkMetal scalar conversion is fixed.
///
/// Set to usize::MAX to completely disable zkMetal.
#[cfg(feature = "zkmetal")]
const GPU_MSM_THRESHOLD: usize = usize::MAX;

/// For ICICLE-only (CPU or GPU), use threshold for efficient dispatch.
#[cfg(all(feature = "icicle", not(feature = "zkmetal")))]
const GPU_MSM_THRESHOLD: usize = 64;

#[cfg(not(any(feature = "icicle", feature = "zkmetal")))]
const GPU_MSM_THRESHOLD: usize = 256;  // Lower threshold for arkworks parallel MSM

/// left[i] = left[i] * scalar + right[i]
fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
    assert_eq!(left.len(), right.len(), "Lengths must match");
    left.par_iter_mut()
        .zip(right.par_iter())
        .for_each(|(l, r)| {
            *l = *l * *scalar + *r;
        });
}

pub struct JoltG1Routines;

impl DoryRoutines<ArkG1> for JoltG1Routines {
    fn msm(bases: &[ArkG1], scalars: &[ArkFr]) -> ArkG1 {
        let len = bases.len();

        // GPU-adaptive dispatch: prefer ICICLE for both CPU and GPU
        #[cfg(feature = "icicle")]
        if len >= GPU_MSM_THRESHOLD {
            let projective_points: &[G1Projective] = unsafe {
                std::slice::from_raw_parts(bases.as_ptr() as *const G1Projective, len)
            };
            let affines = G1Projective::normalize_batch(projective_points);
            let raw_scalars: &[Fr] =
                unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const Fr, len) };
            let result = super::icicle_msm::g1_msm(&affines, raw_scalars);
            return ArkG1(result);
        }

        // Use zkMetal NEON for Apple Silicon
        #[cfg(all(feature = "zkmetal", not(feature = "icicle")))]
        {
            if len >= GPU_MSM_THRESHOLD {
                let projective_points: &[G1Projective] = unsafe {
                    std::slice::from_raw_parts(bases.as_ptr() as *const G1Projective, len)
                };
                let affines = G1Projective::normalize_batch(projective_points);
                let raw_scalars: &[Fr] =
                    unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const Fr, len) };
                let result = super::zkmetal_cpu::pippenger_msm(&affines, raw_scalars);
                return ArkG1(result);
            }
        }

        // Small MSM or no GPU: use arkworks
        let projective_points: &[G1Projective] = unsafe {
            std::slice::from_raw_parts(bases.as_ptr() as *const G1Projective, len)
        };
        let affines = G1Projective::normalize_batch(projective_points);
        let raw_scalars: &[Fr] =
            unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const Fr, len) };

        let result = ArkVariableBaseMSM::msm(&affines, raw_scalars).expect("msm should not fail");
        ArkG1(result)
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG1, scalars: &[ArkFr]) -> Vec<ArkG1> {
        if scalars.is_empty() {
            return vec![];
        }

        // SAFETY: ArkFr has same memory layout as Fr
        let raw_scalars: &[Fr] =
            unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const Fr, scalars.len()) };

        // Note: GLV for G1 requires different precomputed tables than G2
        // Use the standard fixed-base MSM which has its own optimizations
        let results_proj = jolt_optimizations::fixed_base_vector_msm_g1(&base.0, raw_scalars);

        results_proj.into_iter().map(ArkG1).collect()
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG1], vs: &mut [ArkG1], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "bases and vs must have same length");

        // SAFETY: ArkG1 is repr(transparent) so has same memory layout as G1Projective
        let vs_proj: &mut [G1Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G1Projective, vs.len())
        };
        let bases_proj: &[G1Projective] = unsafe {
            std::slice::from_raw_parts(bases.as_ptr() as *const G1Projective, bases.len())
        };

        // SAFETY: ArkFr has same memory layout as Fr
        let raw_scalar = unsafe { std::mem::transmute_copy::<ArkFr, Fr>(scalar) };

        // v[i] = v[i] + scalar * bases[i]
        jolt_optimizations::vector_add_scalar_mul_g1_online(vs_proj, bases_proj, raw_scalar);
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG1], addends: &[ArkG1], scalar: &ArkFr) {
        assert_eq!(
            vs.len(),
            addends.len(),
            "vs and addends must have same length"
        );

        // SAFETY: ArkG1 is repr(transparent) so has same memory layout as G1Projective
        let vs_proj: &mut [G1Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G1Projective, vs.len())
        };
        let addends_proj: &[G1Projective] = unsafe {
            std::slice::from_raw_parts(addends.as_ptr() as *const G1Projective, addends.len())
        };

        // SAFETY: ArkFr has same memory layout as Fr
        let raw_scalar = unsafe { std::mem::transmute_copy::<ArkFr, Fr>(scalar) };

        // v[i] = scalar * v[i] + addends[i]
        jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(
            vs_proj,
            raw_scalar,
            addends_proj,
        );
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        fold_field_vectors(left, right, scalar);
    }
}

pub struct JoltG2Routines;

impl DoryRoutines<ArkG2> for JoltG2Routines {
    fn msm(bases: &[ArkG2], scalars: &[ArkFr]) -> ArkG2 {
        let len = scalars.len();

        // GPU-adaptive dispatch: use ICICLE only for large MSMs
        #[cfg(all(feature = "icicle", not(feature = "zkmetal")))]
        if len >= GPU_MSM_THRESHOLD {
            // SAFETY: ArkG2 is repr(transparent) so has same memory layout as G2Projective
            let projective_points: &[G2Projective] = unsafe {
                std::slice::from_raw_parts(bases.as_ptr() as *const G2Projective, len)
            };
            let affines = G2Projective::normalize_batch(projective_points);

            // SAFETY: ArkFr has same memory layout as Fr
            let raw_scalars: &[Fr] =
                unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const Fr, len) };

            let result = super::icicle_msm::g2_msm(&affines[..len], raw_scalars);
            return ArkG2(result);
        }

        // Small MSM or no GPU: use arkworks
        let projective_points: &[G2Projective] = unsafe {
            std::slice::from_raw_parts(bases.as_ptr() as *const G2Projective, len)
        };
        let affines = G2Projective::normalize_batch(projective_points);
        let raw_scalars: &[Fr] =
            unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const Fr, len) };

        let result = ArkVariableBaseMSM::msm(&affines[..len], raw_scalars)
            .expect("msm should not fail");
        ArkG2(result)
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG2, scalars: &[ArkFr]) -> Vec<ArkG2> {
        if scalars.is_empty() {
            return vec![];
        }

        // SAFETY: ArkFr has same memory layout as Fr
        let raw_scalars: &[Fr] =
            unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const Fr, scalars.len()) };

        // Precompute GLV table ONCE for the fixed base, then reuse for all scalars
        let precomputed = jolt_optimizations::glv_four_precompute(&[base.0]);

        let results_proj: Vec<G2Projective> = raw_scalars
            .par_iter()
            .map(|&scalar| jolt_optimizations::glv_four_scalar_mul(&precomputed, scalar)[0])
            .collect();

        results_proj.into_iter().map(ArkG2).collect()
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG2], vs: &mut [ArkG2], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "bases and vs must have same length");

        // SAFETY: ArkG2 is repr(transparent) so has same memory layout as G2Projective
        let vs_proj: &mut [G2Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G2Projective, vs.len())
        };
        let bases_proj: &[G2Projective] = unsafe {
            std::slice::from_raw_parts(bases.as_ptr() as *const G2Projective, bases.len())
        };

        // SAFETY: ArkFr has same memory layout as Fr
        let raw_scalar = unsafe { std::mem::transmute_copy::<ArkFr, Fr>(scalar) };

        // v[i] = v[i] + scalar * bases[i]
        jolt_optimizations::vector_add_scalar_mul_g2_online(vs_proj, bases_proj, raw_scalar);
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG2], addends: &[ArkG2], scalar: &ArkFr) {
        assert_eq!(
            vs.len(),
            addends.len(),
            "vs and addends must have same length"
        );

        // SAFETY: ArkG2 is repr(transparent) so has same memory layout as G2Projective
        let vs_proj: &mut [G2Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G2Projective, vs.len())
        };
        let addends_proj: &[G2Projective] = unsafe {
            std::slice::from_raw_parts(addends.as_ptr() as *const G2Projective, addends.len())
        };

        // SAFETY: ArkFr has same memory layout as Fr
        let raw_scalar = unsafe { std::mem::transmute_copy::<ArkFr, Fr>(scalar) };

        // v[i] = scalar * v[i] + addends[i]
        jolt_optimizations::vector_scalar_mul_add_gamma_g2_online(
            vs_proj,
            raw_scalar,
            addends_proj,
        );
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        fold_field_vectors(left, right, scalar);
    }
}
