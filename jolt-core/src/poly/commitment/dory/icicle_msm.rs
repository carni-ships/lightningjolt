//! ICICLE-accelerated MSM for BN254 G1 and G2
//!
//! Provides drop-in replacements for arkworks MSM using Ingonyama's ICICLE library.
//! Supports CPU backend by default, Metal GPU backend via `icicle-metal` feature.

use ark_bn254::{Fr, G1Affine as ArkG1Affine, G1Projective, G2Affine as ArkG2Affine, G2Projective};
use ark_ec::AffineRepr;
use ark_ff::PrimeField;
use icicle_bn254::curve::{
    BaseField as IcicleBaseField, G1Affine as IcicleG1Affine, G1Projective as IcicleG1Projective,
    G2Affine as IcicleG2Affine, G2BaseField as IcicleG2BaseField,
    G2Projective as IcicleG2Projective, ScalarField as IcicleScalar,
};
use icicle_core::msm::{MSMConfig, MSM};
use icicle_runtime::memory::HostSlice;
use std::sync::Once;

static INIT: Once = Once::new();

fn ensure_initialized() {
    INIT.call_once(|| {
        // First try Metal backend (macOS GPU), then fall back to CPU
        let result = icicle_runtime::runtime::load_backend("metal");
        if result.is_err() {
            tracing::debug!("ICICLE Metal backend not available, falling back to CPU");
            let _ = icicle_runtime::runtime::load_backend_from_env_or_default();
        } else {
            tracing::debug!("ICICLE Metal backend loaded successfully");
        }

        // Try to set Metal device, fall back to CPU if not available
        let metal_device = icicle_runtime::Device::new("Metal", 0);
        if icicle_runtime::runtime::is_device_available(&metal_device) {
            icicle_runtime::set_device(&metal_device).expect("failed to set Metal device");
            tracing::debug!("ICICLE Using Metal GPU device");
        } else {
            let cpu_device = icicle_runtime::Device::new("CPU", 0);
            icicle_runtime::set_device(&cpu_device).expect("failed to set CPU device");
            tracing::debug!("ICICLE Using CPU device");
        }
    });
}

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
/// arkworks: `[u64; 4]`, ICICLE: `[u32; 8]` — identical bytes on little-endian platforms.
unsafe fn scalars_to_icicle(scalars: &[Fr]) -> &[IcicleScalar] {
    std::slice::from_raw_parts(scalars.as_ptr() as *const IcicleScalar, scalars.len())
}

/// Convert arkworks Fq (Montgomery form) → ICICLE BaseField (standard form).
fn fq_to_icicle_base(fq: &ark_bn254::Fq) -> IcicleBaseField {
    let bigint = fq.into_bigint();
    // SAFETY: BigInt<4> is [u64; 4] = 32 bytes LE, IcicleBaseField is [u32; 8] = 32 bytes LE
    unsafe { std::mem::transmute_copy(&bigint) }
}

/// Convert arkworks Fq2 (Montgomery form) → ICICLE G2BaseField (standard form).
fn fq2_to_icicle_g2base(fq2: &ark_bn254::Fq2) -> IcicleG2BaseField {
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

/// Convert ICICLE BaseField (standard form) → arkworks Fq (Montgomery form).
fn icicle_base_to_fq(base: &IcicleBaseField) -> ark_bn254::Fq {
    // SAFETY: IcicleBaseField is [u32; 8] = 32 bytes LE, BigInt<4> is [u64; 4] = 32 bytes LE
    let bigint: ark_ff::BigInt<4> = unsafe { std::mem::transmute_copy(base) };
    ark_bn254::Fq::from_bigint(bigint).expect("ICICLE base field element out of range")
}

/// Convert ICICLE G2BaseField (standard form) → arkworks Fq2 (Montgomery form).
fn icicle_g2base_to_fq2(base: &IcicleG2BaseField) -> ark_bn254::Fq2 {
    // SAFETY: IcicleG2BaseField is [u32; 16] = 64 bytes LE
    let limbs: [u32; 16] = unsafe { std::mem::transmute_copy(base) };
    let c0_bigint: ark_ff::BigInt<4> =
        unsafe { std::mem::transmute::<[u32; 8], _>(limbs[..8].try_into().unwrap()) };
    let c1_bigint: ark_ff::BigInt<4> =
        unsafe { std::mem::transmute::<[u32; 8], _>(limbs[8..].try_into().unwrap()) };
    let c0 = ark_bn254::Fq::from_bigint(c0_bigint).expect("ICICLE G2 c0 out of range");
    let c1 = ark_bn254::Fq::from_bigint(c1_bigint).expect("ICICLE G2 c1 out of range");
    ark_bn254::Fq2::new(c0, c1)
}

fn g1_affine_to_icicle(points: &[ArkG1Affine]) -> Vec<IcicleG1Affine> {
    points
        .iter()
        .map(|p| {
            if p.is_zero() {
                IcicleG1Affine {
                    x: IcicleBaseField::from(0u32),
                    y: IcicleBaseField::from(0u32),
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

fn g2_affine_to_icicle(points: &[ArkG2Affine]) -> Vec<IcicleG2Affine> {
    points
        .iter()
        .map(|p| {
            if p.is_zero() {
                IcicleG2Affine {
                    x: IcicleG2BaseField::from(0u32),
                    y: IcicleG2BaseField::from(0u32),
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

/// Convert ICICLE G1Projective → arkworks G1Projective via affine intermediate.
///
/// ICICLE uses standard projective (X/Z, Y/Z), arkworks uses Jacobian (X/Z², Y/Z³).
/// Converting through affine avoids coordinate system mismatch.
/// ICICLE returns coordinates in standard form; we convert to Montgomery for arkworks.
fn g1_proj_from_icicle(p: IcicleG1Projective) -> G1Projective {
    let affine: IcicleG1Affine = p.into();
    let zero_affine = IcicleG1Affine {
        x: IcicleBaseField::from(0u32),
        y: IcicleBaseField::from(0u32),
    };
    if affine == zero_affine {
        return G1Projective::default();
    }
    let ark_affine = ark_bn254::G1Affine::new_unchecked(
        icicle_base_to_fq(&affine.x),
        icicle_base_to_fq(&affine.y),
    );
    ark_affine.into()
}

/// Convert ICICLE G2Projective → arkworks G2Projective via affine intermediate.
fn g2_proj_from_icicle(p: IcicleG2Projective) -> G2Projective {
    let affine: IcicleG2Affine = p.into();
    let zero_affine = IcicleG2Affine {
        x: IcicleG2BaseField::from(0u32),
        y: IcicleG2BaseField::from(0u32),
    };
    if affine == zero_affine {
        return G2Projective::default();
    }
    let ark_affine = ark_bn254::G2Affine::new_unchecked(
        icicle_g2base_to_fq2(&affine.x),
        icicle_g2base_to_fq2(&affine.y),
    );
    ark_affine.into()
}

pub fn g1_msm(affines: &[ArkG1Affine], scalars: &[Fr]) -> G1Projective {
    ensure_initialized();

    tracing::debug!(len = affines.len(), "ICICLE G1 MSM GPU-accelerated");

    let icicle_bases = g1_affine_to_icicle(affines);
    // SAFETY: Fr and IcicleScalar have identical 32-byte LE Montgomery layout
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

pub fn g2_msm(affines: &[ArkG2Affine], scalars: &[Fr]) -> G2Projective {
    ensure_initialized();

    tracing::debug!(len = affines.len(), "ICICLE G2 MSM GPU-accelerated");

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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Fr, G1Affine, G1Projective};
    use ark_ec::scalar_mul::variable_base::VariableBaseMSM;
    use ark_ec::CurveGroup;
    use ark_ff::UniformRand;

    #[test]
    fn test_g1_simple_msm() {
        use ark_ec::PrimeGroup;
        use ark_ff::One;

        let generator = G1Projective::generator();
        let gen_affine = generator.into_affine();

        let one = Fr::one();
        let result = g1_msm(&[gen_affine], &[one]);
        assert_eq!(
            result.into_affine(),
            generator.into_affine(),
            "1 * G should equal G"
        );

        let three = Fr::from(3u64);
        let result3 = g1_msm(&[gen_affine], &[three]);
        let expected3 = (generator * three).into_affine();
        assert_eq!(result3.into_affine(), expected3, "3 * G should equal 3G");

        let five = Fr::from(5u64);
        let result8 = g1_msm(&[gen_affine, gen_affine], &[three, five]);
        let expected8 = (generator * Fr::from(8u64)).into_affine();
        assert_eq!(result8.into_affine(), expected8, "(3+5)*G should equal 8G");
    }

    #[test]
    fn test_g1_msm_matches_arkworks() {
        let mut rng = ark_std::test_rng();
        let n = 64;

        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let projectives: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let affines: Vec<G1Affine> = projectives.iter().map(|p| p.into_affine()).collect();

        let ark_result: G1Projective = VariableBaseMSM::msm(&affines, &scalars).unwrap();
        let icicle_result = g1_msm(&affines, &scalars);

        assert_eq!(ark_result, icicle_result, "G1 MSM results must match");
    }

    #[test]
    fn test_g2_msm_matches_arkworks() {
        use ark_bn254::{G2Affine, G2Projective};

        let mut rng = ark_std::test_rng();
        let n = 16;

        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let projectives: Vec<G2Projective> = (0..n).map(|_| G2Projective::rand(&mut rng)).collect();
        let affines: Vec<G2Affine> = projectives.iter().map(|p| p.into_affine()).collect();

        let ark_result: G2Projective = VariableBaseMSM::msm(&affines, &scalars).unwrap();
        let icicle_result = g2_msm(&affines, &scalars);

        assert_eq!(ark_result, icicle_result, "G2 MSM results must match");
    }
}
