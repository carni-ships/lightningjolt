//! ICICLE GPU-accelerated MSM backend for Dory polynomial commitment scheme.
//!
//! This module provides GPU-accelerated multi-scalar multiplication (MSM) implementations
//! using Ingonyama's ICICLE library. It supports both BN254 G1 and G2 curves.
//!
//! ## Features
//!
//! - **GPU acceleration**: Uses ICICLE's optimized MSM implementation
//! - **Multiple backends**: Supports CUDA (with CUDA toolkit) and Metal (macOS)
//! - **Zero-copy scalars**: Leverages memory layout compatibility with arkworks
//!
//! ## Usage
//!
//! Enable the `icicle` feature in your `Cargo.toml`:
//! ```toml
//! dory = { features = ["backends", "icicle"] }
//! ```
//!
//! Then use the ICICLE-backed routines:
//! ```ignore
//! use dory_pcs::backends::icicle_gpu::{IcicleG1Routines, IcicleG2Routines};
//! ```
//!
//! For Metal GPU on macOS:
//! ```toml
//! dory = { features = ["backends", "icicle-metal"] }
//! ```

#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::suspicious_arithmetic_impl)]

use crate::backends::arkworks::{ArkFr, ArkG1, ArkG2};
use crate::primitives::arithmetic::{DoryRoutines, Group};
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
use icicle_core::msm::{MSMConfig, MSM};
use icicle_runtime::memory::HostSlice;
use std::sync::Once;

static INIT: Once = Once::new();

/// Ensure ICICLE runtime is initialized (one-time initialization).
fn ensure_initialized() {
    INIT.call_once(|| {
        let _ = icicle_runtime::runtime::load_backend_from_env_or_default();
        let device = icicle_runtime::Device::new("CPU", 0);
        icicle_runtime::set_device(&device).expect("failed to set ICICLE device");
    });
}

/// Get default MSM configuration optimized for Dory workloads.
///
/// Scalars are passed in Montgomery form (matching arkworks layout).
/// Bases are converted from Montgomery to standard form before MSM.
fn msm_config() -> MSMConfig {
    let mut cfg = MSMConfig::default();
    // Scalars are passed as raw Montgomery-form bytes (zero-copy from arkworks)
    cfg.are_scalars_montgomery_form = true;
    // Bases are explicitly converted to standard form
    cfg.are_bases_montgomery_form = false;
    cfg
}

/// Zero-copy transmute arkworks Fr scalars to ICICLE ScalarField.
///
/// # Safety
/// Both types are 32-byte little-endian Montgomery-form representations of BN254 Fr.
/// arkworks: `[u64; 4]`, ICICLE: `[u32; 8]` - identical bytes on little-endian platforms.
#[inline]
unsafe fn scalars_to_icicle(scalars: &[Fr]) -> &[IcicleScalarField] {
    std::slice::from_raw_parts(scalars.as_ptr() as *const IcicleScalarField, scalars.len())
}

/// Convert arkworks Fq (Montgomery form) to ICICLE base field (standard form).
#[inline]
fn fq_to_icicle_base(fq: &Fq) -> IcicleBaseField {
    let bigint: ark_ff::BigInt<4> = fq.into_bigint();
    // SAFETY: BigInt<4> is [u64; 4] = 32 bytes LE, IcicleBaseField is [u32; 8] = 32 bytes LE
    unsafe { std::mem::transmute_copy(&bigint) }
}

/// Convert arkworks Fq2 (Montgomery form) to ICICLE G2BaseField (standard form).
#[inline]
fn fq2_to_icicle_g2base(fq2: &Fq2) -> IcicleG2BaseField {
    let c0_bigint = fq2.c0.into_bigint();
    let c1_bigint = fq2.c1.into_bigint();
    // SAFETY: two BigInt<4> = 64 bytes LE, IcicleG2BaseField is [u32; 16] = 64 bytes LE
    let mut limbs = [0u32; 16];
    unsafe {
        let c0_u32: [u32; 8] = std::mem::transmute(c0_bigint);
        let c1_u32: [u32; 8] = std::mem::transmute(c1_bigint);
        limbs[..8].copy_from_slice(&c0_u32);
        limbs[8..].copy_from_slice(&c1_u32);
    }
    IcicleG2BaseField::from(limbs)
}

/// Convert ICICLE base field (standard form) to arkworks Fq (Montgomery form).
#[inline]
fn icicle_base_to_fq(base: &IcicleBaseField) -> Fq {
    // SAFETY: IcicleBaseField is [u32; 8] = 32 bytes LE, BigInt<4> is [u64; 4] = 32 bytes LE
    let bigint: ark_ff::BigInt<4> = unsafe { std::mem::transmute_copy(base) };
    Fq::from_bigint(bigint).expect("ICICLE base field element out of range")
}

/// Convert ICICLE G2BaseField (standard form) to arkworks Fq2 (Montgomery form).
#[inline]
fn icicle_g2base_to_fq2(base: &IcicleG2BaseField) -> Fq2 {
    // SAFETY: IcicleG2BaseField is [u32; 16] = 64 bytes LE
    let limbs: [u32; 16] = unsafe { std::mem::transmute_copy(base) };
    let c0_bigint: ark_ff::BigInt<4> =
        unsafe { std::mem::transmute::<[u32; 8], _>(limbs[..8].try_into().unwrap()) };
    let c1_bigint: ark_ff::BigInt<4> =
        unsafe { std::mem::transmute::<[u32; 8], _>(limbs[8..].try_into().unwrap()) };
    let c0 = Fq::from_bigint(c0_bigint).expect("ICICLE G2 c0 out of range");
    let c1 = Fq::from_bigint(c1_bigint).expect("ICICLE G2 c1 out of range");
    Fq2::new(c0, c1)
}

/// Convert a slice of arkworks G1Affine points to ICICLE format.
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

/// Convert a slice of arkworks G2Affine points to ICICLE format.
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

/// Convert ICICLE G1Projective to arkworks G1Projective via affine intermediate.
///
/// ICICLE uses standard projective (X/Z, Y/Z), arkworks uses Jacobian (X/Z^2, Y/Z^3).
/// Converting through affine avoids coordinate system mismatch.
/// ICICLE returns coordinates in standard form; we convert to Montgomery for arkworks.
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
    let ark_affine =
        ArkG1Affine::new_unchecked(icicle_base_to_fq(&affine.x), icicle_base_to_fq(&affine.y));
    ark_affine.into()
}

/// Convert ICICLE G2Projective to arkworks G2Projective via affine intermediate.
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

/// GPU-accelerated G1 MSM using ICICLE.
///
/// # Arguments
/// * `affines` - G1 affine points
/// * `scalars` - Scalar multipliers (in Montgomery form)
fn icicle_g1_msm(affines: &[ArkG1Affine], scalars: &[Fr]) -> ArkG1Projective {
    ensure_initialized();

    let icicle_bases = g1_affine_to_icicle(affines);
    // SAFETY: Fr and IcicleScalarField have identical 32-byte LE Montgomery layout
    let icicle_scalars = unsafe { scalars_to_icicle(scalars) };

    let cfg = msm_config();
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
///
/// # Arguments
/// * `affines` - G2 affine points
/// * `scalars` - Scalar multipliers (in Montgomery form)
fn icicle_g2_msm(affines: &[ArkG2Affine], scalars: &[Fr]) -> ArkG2Projective {
    ensure_initialized();

    let icicle_bases = g2_affine_to_icicle(affines);
    let icicle_scalars = unsafe { scalars_to_icicle(scalars) };

    let cfg = msm_config();
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

/// ICICLE-accelerated DoryRoutines for G1.
///
/// This implementation uses ICICLE's GPU MSM for better performance
/// on large multi-scalar multiplication operations.
pub struct IcicleG1Routines;

impl DoryRoutines<ArkG1> for IcicleG1Routines {
    #[tracing::instrument(skip_all, name = "IcicleG1::msm", fields(len = bases.len()))]
    fn msm(bases: &[ArkG1], scalars: &[ArkFr]) -> ArkG1 {
        assert_eq!(
            bases.len(),
            scalars.len(),
            "MSM requires equal length vectors"
        );

        if bases.is_empty() {
            return ArkG1::identity();
        }

        // Convert ArkG1 to G1Affine
        let affines: Vec<ArkG1Affine> = bases.iter().map(|b| b.0.into_affine()).collect();

        // Extract raw Fr scalars
        let raw_scalars: Vec<Fr> = scalars.iter().map(|s| s.0).collect();

        // Use ICICLE MSM
        ArkG1(icicle_g1_msm(&affines, &raw_scalars))
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG1, scalars: &[ArkFr]) -> Vec<ArkG1> {
        scalars.iter().map(|s| base.scale(s)).collect()
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG1], vs: &mut [ArkG1], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "Lengths must match");

        for (v, base) in vs.iter_mut().zip(bases.iter()) {
            *v = v.add(&base.scale(scalar));
        }
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG1], addends: &[ArkG1], scalar: &ArkFr) {
        assert_eq!(vs.len(), addends.len(), "Lengths must match");

        for (v, addend) in vs.iter_mut().zip(addends.iter()) {
            *v = v.scale(scalar).add(addend);
        }
    }
}

/// ICICLE-accelerated DoryRoutines for G2.
///
/// This implementation uses ICICLE's GPU MSM for better performance
/// on large multi-scalar multiplication operations.
pub struct IcicleG2Routines;

impl DoryRoutines<ArkG2> for IcicleG2Routines {
    #[tracing::instrument(skip_all, name = "IcicleG2::msm", fields(len = bases.len()))]
    fn msm(bases: &[ArkG2], scalars: &[ArkFr]) -> ArkG2 {
        assert_eq!(
            bases.len(),
            scalars.len(),
            "MSM requires equal length vectors"
        );

        if bases.is_empty() {
            return ArkG2::identity();
        }

        // Convert ArkG2 to G2Affine
        let affines: Vec<ArkG2Affine> = bases.iter().map(|b| b.0.into_affine()).collect();

        // Extract raw Fr scalars
        let raw_scalars: Vec<Fr> = scalars.iter().map(|s| s.0).collect();

        // Use ICICLE MSM
        ArkG2(icicle_g2_msm(&affines, &raw_scalars))
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG2, scalars: &[ArkFr]) -> Vec<ArkG2> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            scalars.par_iter().map(|s| base.scale(s)).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            scalars.iter().map(|s| base.scale(s)).collect()
        }
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG2], vs: &mut [ArkG2], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "Lengths must match");

        for (v, base) in vs.iter_mut().zip(bases.iter()) {
            *v = v.add(&base.scale(scalar));
        }
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG2], addends: &[ArkG2], scalar: &ArkFr) {
        assert_eq!(vs.len(), addends.len(), "Lengths must match");

        for (v, addend) in vs.iter_mut().zip(addends.iter()) {
            *v = v.scale(scalar).add(addend);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::scalar_mul::variable_base::VariableBaseMSM;
    use ark_ec::CurveGroup;
    use ark_ff::UniformRand;

    #[test]
    fn test_icicle_g1_simple_msm() {
        use ark_ec::PrimeGroup;
        use ark_ff::One;

        let generator = ArkG1Projective::generator();
        let gen_affine = generator.into_affine();

        let one = Fr::one();
        let result = icicle_g1_msm(&[gen_affine], &[one]);
        assert_eq!(
            result.into_affine(),
            generator.into_affine(),
            "1 * G should equal G"
        );

        let three = Fr::from(3u64);
        let result3 = icicle_g1_msm(&[gen_affine], &[three]);
        let expected3 = (generator * three).into_affine();
        assert_eq!(result3.into_affine(), expected3, "3 * G should equal 3G");

        let five = Fr::from(5u64);
        let result8 = icicle_g1_msm(&[gen_affine, gen_affine], &[three, five]);
        let expected8 = (generator * Fr::from(8u64)).into_affine();
        assert_eq!(result8.into_affine(), expected8, "(3+5)*G should equal 8G");
    }

    #[test]
    fn test_icicle_g1_msm_matches_arkworks() {
        let mut rng = ark_std::test_rng();
        let n = 64;

        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let projectives: Vec<ArkG1Projective> = (0..n).map(|_| ArkG1Projective::rand(&mut rng)).collect();
        let affines: Vec<ArkG1Affine> = projectives.iter().map(|p| p.into_affine()).collect();

        let ark_result: ArkG1Projective = VariableBaseMSM::msm(&affines, &scalars).unwrap();
        let icicle_result = icicle_g1_msm(&affines, &scalars);

        assert_eq!(ark_result, icicle_result, "G1 MSM results must match");
    }

    #[test]
    fn test_icicle_g2_msm_matches_arkworks() {
        let mut rng = ark_std::test_rng();
        let n = 16;

        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let projectives: Vec<ArkG2Projective> = (0..n).map(|_| ArkG2Projective::rand(&mut rng)).collect();
        let affines: Vec<ArkG2Affine> = projectives.iter().map(|p| p.into_affine()).collect();

        let ark_result: ArkG2Projective = VariableBaseMSM::msm(&affines, &scalars).unwrap();
        let icicle_result = icicle_g2_msm(&affines, &scalars);

        assert_eq!(ark_result, icicle_result, "G2 MSM results must match");
    }
}
