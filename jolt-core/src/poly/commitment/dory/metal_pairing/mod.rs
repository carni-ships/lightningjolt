//! Metal GPU-accelerated BN254 pairing for Apple Silicon
//!
//! Provides GPU-accelerated multi-pairing using Metal compute shaders.
//! Feature-gated behind `metal-pairing`.

mod gpu;

#[cfg(test)]
mod tests;

use ark_bn254::{Fq, Fq12, Fq2, Fq6, G1Affine, G2Affine};
use ark_ff::One;
use gpu::{Fq12Limbs, G1AffineLimbs, G2AffineLimbs};
use rayon::prelude::*;
use std::sync::Once;

static REGISTER_ONCE: Once = Once::new();

/// Register the Metal GPU Miller loop with the dory-pcs pairing backend.
/// Safe to call multiple times; only the first call has effect.
pub(crate) fn register() {
    REGISTER_ONCE.call_once(|| {
        dory::backends::arkworks::register_gpu_batch_miller_loop(metal_batch_miller_loop);
    });
}

/// GPU batch Miller loop: computes ∏ ML(g1[i], g2[i]).
fn metal_batch_miller_loop(g1: &[G1Affine], g2: &[G2Affine]) -> Fq12 {
    if g1.is_empty() {
        return Fq12::one();
    }

    let g1_limbs: Vec<G1AffineLimbs> = g1.iter().map(g1_affine_to_limbs).collect();
    let g2_limbs: Vec<G2AffineLimbs> = g2.iter().map(g2_affine_to_limbs).collect();

    let result_limbs = gpu::gpu_miller_loop(&g1_limbs, &g2_limbs);

    // Parallel conversion + tree reduction of Fq12 products
    parallel_fq12_product(&result_limbs)
}

/// Parallel tree reduction: convert limbs to Fq12, then reduce via parallel chunks.
fn parallel_fq12_product(limbs: &[Fq12Limbs]) -> Fq12 {
    if limbs.is_empty() {
        return Fq12::one();
    }
    // Parallel conversion
    let mut vals: Vec<Fq12> = limbs.par_iter().map(limbs_to_fq12).collect();
    // Tree reduction
    while vals.len() > 1 {
        vals = vals
            .par_chunks(2)
            .map(|chunk| {
                if chunk.len() == 2 {
                    chunk[0] * chunk[1]
                } else {
                    chunk[0]
                }
            })
            .collect();
    }
    vals[0]
}

fn fq_to_limbs(f: &Fq) -> [u32; 8] {
    let mont_limbs: [u64; 4] = f.0 .0;
    let mut result = [0u32; 8];
    for i in 0..4 {
        result[2 * i] = mont_limbs[i] as u32;
        result[2 * i + 1] = (mont_limbs[i] >> 32) as u32;
    }
    result
}

fn limbs_to_fq(limbs: &[u32; 8]) -> Fq {
    let mut u64_limbs = [0u64; 4];
    for i in 0..4 {
        u64_limbs[i] = limbs[2 * i] as u64 | ((limbs[2 * i + 1] as u64) << 32);
    }
    Fq::new_unchecked(ark_ff::BigInt(u64_limbs))
}

fn g1_affine_to_limbs(p: &G1Affine) -> G1AffineLimbs {
    let mut limbs = [0u32; 16];
    limbs[..8].copy_from_slice(&fq_to_limbs(&p.x));
    limbs[8..].copy_from_slice(&fq_to_limbs(&p.y));
    limbs
}

fn g2_affine_to_limbs(p: &G2Affine) -> G2AffineLimbs {
    let mut limbs = [0u32; 32];
    limbs[..8].copy_from_slice(&fq_to_limbs(&p.x.c0));
    limbs[8..16].copy_from_slice(&fq_to_limbs(&p.x.c1));
    limbs[16..24].copy_from_slice(&fq_to_limbs(&p.y.c0));
    limbs[24..].copy_from_slice(&fq_to_limbs(&p.y.c1));
    limbs
}

fn limbs_to_fq12(limbs: &Fq12Limbs) -> Fq12 {
    let mut fq_elems = [Fq::default(); 12];
    for (i, elem) in fq_elems.iter_mut().enumerate() {
        let chunk: [u32; 8] = limbs[i * 8..(i + 1) * 8].try_into().unwrap();
        *elem = limbs_to_fq(&chunk);
    }
    Fq12::new(
        Fq6::new(
            Fq2::new(fq_elems[0], fq_elems[1]),
            Fq2::new(fq_elems[2], fq_elems[3]),
            Fq2::new(fq_elems[4], fq_elems[5]),
        ),
        Fq6::new(
            Fq2::new(fq_elems[6], fq_elems[7]),
            Fq2::new(fq_elems[8], fq_elems[9]),
            Fq2::new(fq_elems[10], fq_elems[11]),
        ),
    )
}
