//! BN254 pairing implementation with optimizations

#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use super::ark_group::{ArkG1, ArkG2, ArkGT};
use crate::primitives::arithmetic::{Group, PairingCurve};
use ark_bn254::Bn254;
use ark_ec::pairing::Pairing;

use ark_ff::One;

use std::sync::OnceLock;

/// Callback type for GPU-accelerated batch Miller loop.
/// Takes affine G1 and G2 points, returns the product of individual Miller loop outputs as Fq12.
pub type GpuBatchMillerLoopFn =
    fn(&[ark_bn254::G1Affine], &[ark_bn254::G2Affine]) -> ark_bn254::Fq12;

static GPU_BATCH_MILLER_LOOP: OnceLock<GpuBatchMillerLoopFn> = OnceLock::new();

const GPU_MIN_BATCH_SIZE: usize = 4096;

/// Register a GPU-accelerated batch Miller loop function.
/// Once registered, parallel multi-pairing functions will use it instead of arkworks.
pub fn register_gpu_batch_miller_loop(f: GpuBatchMillerLoopFn) {
    let _ = GPU_BATCH_MILLER_LOOP.set(f);
}

#[derive(Default, Clone, Debug)]
pub struct BN254;

mod pairing_helpers {
    use super::*;
    use super::{ArkG1, ArkG2, ArkGT};

    /// Determine optimal chunk size for parallel Miller loop computation.
    /// Tuned: larger chunks reduce per-chunk overhead (affine prep + Miller loop startup).
    #[cfg(feature = "parallel")]
    fn determine_chunk_size(total: usize) -> usize {
        // Larger minimum chunk: Miller loop startup cost is significant,
        // so we want fewer, bigger chunks. Each chunk runs one multi_miller_loop.
        const MIN_CHUNK: usize = 128;
        const MAX_CHUNK: usize = 1024;

        if total < MIN_CHUNK {
            return total;
        }

        let num_threads = rayon::current_num_threads();
        let chunk = total.div_ceil(num_threads);
        chunk.clamp(MIN_CHUNK, MAX_CHUNK)
    }

    /// Batch convert G1 projective points to affine using Montgomery's trick.
    /// Uses 1 field inversion + 3(N-1) multiplications instead of N inversions.
    #[cfg(feature = "parallel")]
    pub(super) fn batch_g1_to_affine(points: &[ArkG1]) -> Vec<ark_bn254::G1Affine> {
        use ark_bn254::{Fq, G1Affine, G1Projective};
        use ark_ff::{Field, Zero};

        let n = points.len();
        if n == 0 {
            return vec![];
        }

        // Filter out identity points and track indices
        let mut results = vec![G1Affine::identity(); n];
        let mut non_zero_indices = Vec::with_capacity(n);
        let mut z_values = Vec::with_capacity(n);

        for (i, p) in points.iter().enumerate() {
            let proj: G1Projective = p.0;
            if proj.z.is_zero() {
                // Identity — already set to identity in results
            } else {
                non_zero_indices.push(i);
                z_values.push(proj.z);
            }
        }

        if z_values.is_empty() {
            return results;
        }

        // Montgomery's batch inversion: compute all 1/z_i with a single inversion
        let mut products = vec![Fq::ONE; z_values.len()];
        products[0] = z_values[0];
        for i in 1..z_values.len() {
            products[i] = products[i - 1] * z_values[i];
        }

        // Single inversion of the accumulated product
        let mut inv = products[z_values.len() - 1]
            .inverse()
            .expect("accumulated Z product should be nonzero");

        // Back-propagate to get individual inverses
        let mut z_invs = vec![Fq::ONE; z_values.len()];
        for i in (1..z_values.len()).rev() {
            z_invs[i] = inv * products[i - 1];
            inv *= z_values[i];
        }
        z_invs[0] = inv;

        // Compute affine coordinates: x = X * z_inv^2, y = Y * z_inv^3
        for (j, &idx) in non_zero_indices.iter().enumerate() {
            let proj: G1Projective = points[idx].0;
            let zi = z_invs[j];
            let zi2 = zi * zi;
            let zi3 = zi2 * zi;
            results[idx] = G1Affine::new_unchecked(proj.x * zi2, proj.y * zi3);
        }

        results
    }

    /// Batch convert G2 projective points to affine using Montgomery's trick.
    #[cfg(feature = "parallel")]
    pub(super) fn batch_g2_to_affine(points: &[ArkG2]) -> Vec<ark_bn254::G2Affine> {
        use ark_bn254::{Fq2, G2Affine, G2Projective};
        use ark_ff::{Field, Zero};

        let n = points.len();
        if n == 0 {
            return vec![];
        }

        let mut results = vec![G2Affine::identity(); n];
        let mut non_zero_indices = Vec::with_capacity(n);
        let mut z_values = Vec::with_capacity(n);

        for (i, p) in points.iter().enumerate() {
            let proj: G2Projective = p.0;
            if proj.z.is_zero() {
                // Identity
            } else {
                non_zero_indices.push(i);
                z_values.push(proj.z);
            }
        }

        if z_values.is_empty() {
            return results;
        }

        let mut products = vec![Fq2::ONE; z_values.len()];
        products[0] = z_values[0];
        for i in 1..z_values.len() {
            products[i] = products[i - 1] * z_values[i];
        }

        let mut inv = products[z_values.len() - 1]
            .inverse()
            .expect("accumulated Z product should be nonzero");

        let mut z_invs = vec![Fq2::ONE; z_values.len()];
        for i in (1..z_values.len()).rev() {
            z_invs[i] = inv * products[i - 1];
            inv *= z_values[i];
        }
        z_invs[0] = inv;

        for (j, &idx) in non_zero_indices.iter().enumerate() {
            let proj: G2Projective = points[idx].0;
            let zi = z_invs[j];
            let zi2 = zi * zi;
            let zi3 = zi2 * zi;
            results[idx] = G2Affine::new_unchecked(proj.x * zi2, proj.y * zi3);
        }

        results
    }

    /// Sequential multi-pairing
    #[allow(dead_code)]
    #[tracing::instrument(skip_all, name = "multi_pair_sequential", fields(len = ps.len()))]
    pub(super) fn multi_pair_sequential(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        use ark_bn254::{G1Affine, G2Affine};

        let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> = ps
            .iter()
            .map(|p| {
                let affine: G1Affine = p.0.into();
                affine.into()
            })
            .collect();

        let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> = qs
            .iter()
            .map(|q| {
                let affine: G2Affine = q.0.into();
                affine.into()
            })
            .collect();

        multi_pair_with_prepared(ps_prep, &qs_prep)
    }

    /// Sequential multi-pairing with G2 from setup (uses cache if available)
    #[allow(dead_code)]
    #[tracing::instrument(skip_all, name = "multi_pair_g2_setup_sequential", fields(len = ps.len()))]
    pub(super) fn multi_pair_g2_setup_sequential(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        {
            use ark_bn254::G1Affine;

            let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> = ps
                .iter()
                .map(|p| {
                    let affine: G1Affine = p.0.into();
                    affine.into()
                })
                .collect();

            #[cfg(feature = "cache")]
            {
                if let Some(cache) = crate::backends::arkworks::ark_cache::get_prepared_cache() {
                    return multi_pair_with_prepared(ps_prep, &cache.g2_prepared[..qs.len()]);
                }
            }

            use ark_bn254::G2Affine;
            let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> = qs
                .iter()
                .map(|q| {
                    let affine: G2Affine = q.0.into();
                    affine.into()
                })
                .collect();
            multi_pair_with_prepared(ps_prep, &qs_prep)
        }
    }

    /// Sequential multi-pairing with G1 from setup (uses cache if available)
    #[allow(dead_code)]
    #[tracing::instrument(skip_all, name = "multi_pair_g1_setup_sequential", fields(len = ps.len()))]
    pub(super) fn multi_pair_g1_setup_sequential(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        use ark_bn254::G2Affine;

        let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> = qs
            .iter()
            .map(|q| {
                let affine: G2Affine = q.0.into();
                affine.into()
            })
            .collect();

        #[cfg(feature = "cache")]
        {
            if let Some(cache) = crate::backends::arkworks::ark_cache::get_prepared_cache() {
                let ps_prep: Vec<_> = ps
                    .iter()
                    .enumerate()
                    .map(|(i, _)| cache.g1_prepared[i].clone())
                    .collect();
                return multi_pair_with_prepared(ps_prep, &qs_prep);
            }
        }

        use ark_bn254::G1Affine;
        let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> = ps
            .iter()
            .map(|p| {
                let affine: G1Affine = p.0.into();
                affine.into()
            })
            .collect();
        multi_pair_with_prepared(ps_prep, &qs_prep)
    }

    fn multi_pair_with_prepared(
        ps_prep: Vec<<Bn254 as Pairing>::G1Prepared>,
        qs_prep: &[<Bn254 as Pairing>::G2Prepared],
    ) -> ArkGT {
        let miller_output = multi_miller_loop_single_acc(&ps_prep, qs_prep);
        let result = Bn254::final_exponentiation(miller_output)
            .expect("Final exponentiation should not fail");
        ArkGT(result.0)
    }

    /// Single-accumulator Miller loop for BN254.
    /// Arkworks' `multi_miller_loop` splits pairs into chunks of 4 with separate
    /// accumulators, causing redundant Fq12 squarings. For N pairs with chunk size 4:
    /// ceil(N/4) × 63 squarings + (ceil(N/4)-1) product muls.
    /// This uses one accumulator: always 63 squarings regardless of N.
    fn multi_miller_loop_single_acc(
        ps: &[<Bn254 as Pairing>::G1Prepared],
        qs: &[<Bn254 as Pairing>::G2Prepared],
    ) -> ark_ec::pairing::MillerLoopOutput<Bn254> {
        use ark_bn254::Config;
        use ark_ec::models::bn::BnConfig;
        use ark_ff::Field;

        let mut pairs: Vec<_> = ps
            .iter()
            .zip(qs.iter())
            .filter(|(p, q)| !p.is_zero() && !q.infinity)
            .map(|(p, q)| (p.0, q.ell_coeffs.iter()))
            .collect();

        if pairs.is_empty() {
            return ark_ec::pairing::MillerLoopOutput(<<Bn254 as Pairing>::TargetField>::one());
        }

        let mut f = <<Bn254 as Pairing>::TargetField>::one();

        for i in (1..Config::ATE_LOOP_COUNT.len()).rev() {
            if i != Config::ATE_LOOP_COUNT.len() - 1 {
                f.square_in_place();
            }

            for (p, coeffs) in pairs.iter_mut() {
                ell_bn254_d_twist(&mut f, coeffs.next().unwrap(), p);
            }

            let bit = Config::ATE_LOOP_COUNT[i - 1];
            if bit == 1 || bit == -1 {
                for (p, coeffs) in pairs.iter_mut() {
                    ell_bn254_d_twist(&mut f, coeffs.next().unwrap(), p);
                }
            }
        }

        // BN254: X_IS_NEGATIVE = false, no cyclotomic_inverse needed

        for (p, coeffs) in &mut pairs {
            ell_bn254_d_twist(&mut f, coeffs.next().unwrap(), p);
        }

        for (p, coeffs) in &mut pairs {
            ell_bn254_d_twist(&mut f, coeffs.next().unwrap(), p);
        }

        ark_ec::pairing::MillerLoopOutput(f)
    }

    /// BN254 D-twist line evaluation: multiply Fq12 accumulator by sparse line coefficients.
    #[inline(always)]
    fn ell_bn254_d_twist(
        f: &mut ark_bn254::Fq12,
        coeffs: &(ark_bn254::Fq2, ark_bn254::Fq2, ark_bn254::Fq2),
        p: &ark_bn254::G1Affine,
    ) {
        let mut c0 = coeffs.0;
        let mut c1 = coeffs.1;
        let c2 = coeffs.2;
        c0.mul_assign_by_fp(&p.y);
        c1.mul_assign_by_fp(&p.x);
        f.mul_by_034(&c0, &c1, &c2);
    }

    /// Parallel multi-pairing with batch affine conversion and chunked Miller loops.
    #[cfg(feature = "parallel")]
    #[tracing::instrument(skip_all, name = "multi_pair_parallel", fields(len = ps.len()))]
    pub(super) fn multi_pair_parallel(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        use rayon::prelude::*;

        let n = ps.len();

        // Batch convert all G1 and G2 to affine first (single batch inversion each)
        let ps_affine = batch_g1_to_affine(ps);
        let qs_affine = batch_g2_to_affine(qs);

        // GPU fast-path: only use GPU for large batches where it can amortize dispatch overhead
        if n >= GPU_MIN_BATCH_SIZE {
            if let Some(gpu_fn) = GPU_BATCH_MILLER_LOOP.get() {
                let combined_fq12 = gpu_fn(&ps_affine, &qs_affine);
                let combined = ark_ec::pairing::MillerLoopOutput(combined_fq12);
                let result = Bn254::final_exponentiation(combined)
                    .expect("Final exponentiation should not fail");
                return ArkGT(result.0);
            }
        }

        // For small inputs, skip chunking overhead
        if n <= 128 {
            let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> =
                ps_affine.into_iter().map(|a| a.into()).collect();
            let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> =
                qs_affine.into_iter().map(|a| a.into()).collect();
            return multi_pair_with_prepared(ps_prep, &qs_prep);
        }

        let chunk_size = determine_chunk_size(n);

        let combined = ps_affine
            .par_chunks(chunk_size)
            .zip(qs_affine.par_chunks(chunk_size))
            .map(|(ps_chunk, qs_chunk)| {
                let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> =
                    ps_chunk.iter().map(|&a| a.into()).collect();
                let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> =
                    qs_chunk.iter().map(|&a| a.into()).collect();
                multi_miller_loop_single_acc(&ps_prep, &qs_prep)
            })
            .reduce(
                || ark_ec::pairing::MillerLoopOutput(<<Bn254 as Pairing>::TargetField>::one()),
                |a, b| ark_ec::pairing::MillerLoopOutput(a.0 * b.0),
            );

        let result =
            Bn254::final_exponentiation(combined).expect("Final exponentiation should not fail");
        ArkGT(result.0)
    }

    /// Parallel multi-pairing with G2 from setup (uses cache for G2, batch affine for G1).
    /// Skips identity G1 points to avoid wasted Miller loop iterations on sparse inputs.
    #[cfg(feature = "parallel")]
    #[tracing::instrument(skip_all, name = "multi_pair_g2_setup_parallel", fields(len = ps.len()))]
    pub(super) fn multi_pair_g2_setup_parallel(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        use ark_ec::AffineRepr;

        // Batch convert all G1 points to affine (Montgomery's trick — 1 inversion total)
        let ps_affine = batch_g1_to_affine(ps);

        // Filter out identity G1 points, preserving original indices for G2 cache lookup
        let non_zero: Vec<(usize, ark_bn254::G1Affine)> = ps_affine
            .into_iter()
            .enumerate()
            .filter(|(_, a)| !a.is_zero())
            .collect();

        if non_zero.is_empty() {
            return ArkGT(<<Bn254 as Pairing>::TargetField>::one());
        }

        {
            use rayon::prelude::*;
            // GPU fast-path: only use GPU for large batches
            if non_zero.len() >= GPU_MIN_BATCH_SIZE {
                if let Some(gpu_fn) = GPU_BATCH_MILLER_LOOP.get() {
                    let g1_affines: Vec<ark_bn254::G1Affine> =
                        non_zero.iter().map(|(_, a)| *a).collect();
                    let g2_affines: Vec<ark_bn254::G2Affine> = non_zero
                        .iter()
                        .map(|(orig_idx, _)| qs[*orig_idx].0.into())
                        .collect();
                    let combined_fq12 = gpu_fn(&g1_affines, &g2_affines);
                    let combined = ark_ec::pairing::MillerLoopOutput(combined_fq12);
                    let result = Bn254::final_exponentiation(combined)
                        .expect("Final exponentiation should not fail");
                    return ArkGT(result.0);
                }
            }

            // Fast path: if no identity points were filtered out, indices are 0..n
            // and we can slice the G2 cache directly without cloning.
            #[cfg(feature = "cache")]
            let contiguous = non_zero.len() == ps.len();
            #[cfg(not(feature = "cache"))]
            let contiguous = false;

            let chunk_size = determine_chunk_size(non_zero.len());

            #[cfg(feature = "cache")]
            let cache = crate::backends::arkworks::ark_cache::get_prepared_cache();

            let combined = if contiguous {
                #[cfg(feature = "cache")]
                {
                    if let Some(ref c) = cache {
                        let g1_affines: Vec<ark_bn254::G1Affine> =
                            non_zero.iter().map(|(_, a)| *a).collect();
                        let g2_prep = &c.g2_prepared[..non_zero.len()];

                        g1_affines
                            .par_chunks(chunk_size)
                            .zip(g2_prep.par_chunks(chunk_size))
                            .map(|(g1_chunk, g2_chunk)| {
                                let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> =
                                    g1_chunk.iter().map(|a| (*a).into()).collect();
                                multi_miller_loop_single_acc(&ps_prep, g2_chunk)
                            })
                            .reduce(
                                || {
                                    ark_ec::pairing::MillerLoopOutput(
                                        <<Bn254 as Pairing>::TargetField>::one(),
                                    )
                                },
                                |a, b| ark_ec::pairing::MillerLoopOutput(a.0 * b.0),
                            )
                    } else {
                        // Cache not initialized, fall through to clone path
                        non_zero
                            .par_chunks(chunk_size)
                            .map(|chunk| {
                                let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> =
                                    chunk.iter().map(|(_, a)| (*a).into()).collect();
                                let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> = {
                                    use ark_bn254::G2Affine;
                                    chunk
                                        .iter()
                                        .map(|(orig_idx, _)| {
                                            let affine: G2Affine = qs[*orig_idx].0.into();
                                            affine.into()
                                        })
                                        .collect()
                                };
                                multi_miller_loop_single_acc(&ps_prep, &qs_prep)
                            })
                            .reduce(
                                || {
                                    ark_ec::pairing::MillerLoopOutput(
                                        <<Bn254 as Pairing>::TargetField>::one(),
                                    )
                                },
                                |a, b| ark_ec::pairing::MillerLoopOutput(a.0 * b.0),
                            )
                    }
                }
                #[cfg(not(feature = "cache"))]
                unreachable!()
            } else {
                non_zero
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> =
                            chunk.iter().map(|(_, a)| (*a).into()).collect();

                        #[cfg(feature = "cache")]
                        let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> =
                            if let Some(ref c) = cache {
                                chunk
                                    .iter()
                                    .map(|(orig_idx, _)| c.g2_prepared[*orig_idx].clone())
                                    .collect()
                            } else {
                                use ark_bn254::G2Affine;
                                chunk
                                    .iter()
                                    .map(|(orig_idx, _)| {
                                        let affine: G2Affine = qs[*orig_idx].0.into();
                                        affine.into()
                                    })
                                    .collect()
                            };
                        #[cfg(not(feature = "cache"))]
                        let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> = {
                            use ark_bn254::G2Affine;
                            chunk
                                .iter()
                                .map(|(orig_idx, _)| {
                                    let affine: G2Affine = qs[*orig_idx].0.into();
                                    affine.into()
                                })
                                .collect()
                        };

                        multi_miller_loop_single_acc(&ps_prep, &qs_prep)
                    })
                    .reduce(
                        || {
                            ark_ec::pairing::MillerLoopOutput(
                                <<Bn254 as Pairing>::TargetField>::one(),
                            )
                        },
                        |a, b| ark_ec::pairing::MillerLoopOutput(a.0 * b.0),
                    )
            };

            let result = Bn254::final_exponentiation(combined)
                .expect("Final exponentiation should not fail");
            ArkGT(result.0)
        }
    }

    /// Parallel multi-pairing with G1 from setup (uses cache for G1, batch affine for G2)
    #[cfg(feature = "parallel")]
    #[tracing::instrument(skip_all, name = "multi_pair_g1_setup_parallel", fields(len = ps.len()))]
    pub(super) fn multi_pair_g1_setup_parallel(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        {
            use rayon::prelude::*;
            let n = qs.len();
            // Batch convert all G2 points to affine
            let qs_affine = batch_g2_to_affine(qs);

            // GPU fast-path: only use GPU for large batches
            if n >= GPU_MIN_BATCH_SIZE {
                if let Some(gpu_fn) = GPU_BATCH_MILLER_LOOP.get() {
                    let ps_affine = batch_g1_to_affine(ps);
                    let combined_fq12 = gpu_fn(&ps_affine, &qs_affine);
                    let combined = ark_ec::pairing::MillerLoopOutput(combined_fq12);
                    let result = Bn254::final_exponentiation(combined)
                        .expect("Final exponentiation should not fail");
                    return ArkGT(result.0);
                }
            }

            let chunk_size = determine_chunk_size(n);

            #[cfg(feature = "cache")]
            let cache = crate::backends::arkworks::ark_cache::get_prepared_cache();

            let combined = qs_affine
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(chunk_idx, qs_chunk)| {
                    let start_idx = chunk_idx * chunk_size;
                    let end_idx = start_idx + qs_chunk.len();

                    let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> =
                        qs_chunk.iter().map(|&a| a.into()).collect();

                    #[cfg(feature = "cache")]
                    let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> = if let Some(ref c) = cache {
                        c.g1_prepared[start_idx..end_idx].to_vec()
                    } else {
                        use ark_bn254::G1Affine;
                        ps[start_idx..end_idx]
                            .iter()
                            .map(|p| {
                                let affine: G1Affine = p.0.into();
                                affine.into()
                            })
                            .collect()
                    };
                    #[cfg(not(feature = "cache"))]
                    let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> = {
                        use ark_bn254::G1Affine;
                        ps[start_idx..end_idx]
                            .iter()
                            .map(|p| {
                                let affine: G1Affine = p.0.into();
                                affine.into()
                            })
                            .collect()
                    };

                    multi_miller_loop_single_acc(&ps_prep, &qs_prep)
                })
                .reduce(
                    || ark_ec::pairing::MillerLoopOutput(<<Bn254 as Pairing>::TargetField>::one()),
                    |a, b| ark_ec::pairing::MillerLoopOutput(a.0 * b.0),
                );

            let result = Bn254::final_exponentiation(combined)
                .expect("Final exponentiation should not fail");
            ArkGT(result.0)
        }
    }

    /// Optimized multi-pairing dispatch
    pub(super) fn multi_pair_optimized(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        #[cfg(feature = "parallel")]
        {
            multi_pair_parallel(ps, qs)
        }
        #[cfg(not(feature = "parallel"))]
        {
            multi_pair_sequential(ps, qs)
        }
    }

    /// Optimized multi-pairing dispatch for G2 from setup
    pub(super) fn multi_pair_g2_setup_optimized(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        #[cfg(feature = "parallel")]
        {
            multi_pair_g2_setup_parallel(ps, qs)
        }
        #[cfg(not(feature = "parallel"))]
        {
            multi_pair_g2_setup_sequential(ps, qs)
        }
    }

    /// Optimized multi-pairing dispatch for G1 from setup
    pub(super) fn multi_pair_g1_setup_optimized(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        #[cfg(feature = "parallel")]
        {
            multi_pair_g1_setup_parallel(ps, qs)
        }
        #[cfg(not(feature = "parallel"))]
        {
            multi_pair_g1_setup_sequential(ps, qs)
        }
    }

    /// Batch pairing returning individual GT results (single Miller loop, separate final exp).
    ///
    /// This is more efficient than calling multi_pair for each pair when we need
    /// individual results (e.g., when applying different masks to each result).
    #[cfg(feature = "parallel")]
    #[tracing::instrument(skip_all, name = "multi_pair_batch_individual", fields(len = ps.len()))]
    pub(super) fn multi_pair_batch_individual(ps: &[ArkG1], qs: &[ArkG2]) -> Vec<ArkGT> {
        use rayon::prelude::*;

        let n = ps.len();
        if n == 0 {
            return vec![];
        }

        // Batch convert all G1 and G2 to affine first (single batch inversion each)
        let ps_affine = batch_g1_to_affine(ps);
        let qs_affine = batch_g2_to_affine(qs);

        let chunk_size = determine_chunk_size(n);

        // For small inputs, just compute individually
        if n <= 64 {
            return ps_affine
                .iter()
                .zip(qs_affine.iter())
                .map(|(&p, &q)| {
                    let p_prep: <Bn254 as Pairing>::G1Prepared = p.into();
                    let q_prep: <Bn254 as Pairing>::G2Prepared = q.into();
                    let result = Bn254::final_exponentiation(
                        Bn254::miller_loop(p_prep, q_prep),
                    )
                    .expect("Final exponentiation should not fail");
                    ArkGT(result.0)
                })
                .collect();
        }

        // Parallel Miller loops for each chunk, then final exponentiation per result
        ps_affine
            .par_chunks(chunk_size)
            .zip(qs_affine.par_chunks(chunk_size))
            .flat_map_iter(|(ps_chunk, qs_chunk)| {
                let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> =
                    ps_chunk.iter().map(|&a| a.into()).collect();
                let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> =
                    qs_chunk.iter().map(|&a| a.into()).collect();

                ps_prep
                    .into_iter()
                    .zip(qs_prep.into_iter())
                    .map(|(p_prep, q_prep)| {
                        let result = Bn254::final_exponentiation(
                            Bn254::miller_loop(p_prep, q_prep),
                        )
                        .expect("Final exponentiation should not fail");
                        ArkGT(result.0)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Sequential version of multi_pair_batch_individual
    #[cfg(not(feature = "parallel"))]
    pub(super) fn multi_pair_batch_individual(ps: &[ArkG1], qs: &[ArkG2]) -> Vec<ArkGT> {
        use ark_bn254::{G1Affine, G2Affine};

        let n = ps.len();
        if n == 0 {
            return vec![];
        }

        ps.iter()
            .zip(qs.iter())
            .map(|(p, q)| {
                let p_affine: G1Affine = p.0.into();
                let q_affine: G2Affine = q.0.into();
                let p_prep: <Bn254 as Pairing>::G1Prepared = p_affine.into();
                let q_prep: <Bn254 as Pairing>::G2Prepared = q_affine.into();
                let result = Bn254::final_exponentiation(
                    Bn254::miller_loop(p_prep, q_prep),
                )
                .expect("Final exponentiation should not fail");
                ArkGT(result.0)
            })
            .collect()
    }

    /// Batch pairing with G2 from setup, returning individual GT results.
    #[cfg(feature = "parallel")]
    pub(super) fn multi_pair_g2_setup_batch_individual(ps: &[ArkG1], qs: &[ArkG2]) -> Vec<ArkGT> {
        use ark_ec::AffineRepr;
        use rayon::prelude::*;

        // Batch convert all G1 points to affine (Montgomery's trick — 1 inversion total)
        let ps_affine = batch_g1_to_affine(ps);

        // Filter out identity G1 points, preserving original indices for G2 cache lookup
        let non_zero: Vec<(usize, ark_bn254::G1Affine)> = ps_affine
            .into_iter()
            .enumerate()
            .filter(|(_, a)| !a.is_zero())
            .collect();

        if non_zero.is_empty() {
            return vec![ArkGT(<<Bn254 as Pairing>::TargetField>::one()); ps.len()];
        }

        let chunk_size = determine_chunk_size(non_zero.len());

        #[cfg(feature = "cache")]
        let cache = crate::backends::arkworks::ark_cache::get_prepared_cache();

        non_zero
            .par_chunks(chunk_size)
            .flat_map_iter(|chunk| {
                let results: Vec<ArkGT> = chunk
                    .iter()
                    .map(|(orig_idx, g1_affine)| {
                        let g2_affine: ark_bn254::G2Affine = qs[*orig_idx].0.into();
                        let g2_prep: <Bn254 as Pairing>::G2Prepared = g2_affine.into();
                        let g1_prep: <Bn254 as Pairing>::G1Prepared = (*g1_affine).into();

                        #[cfg(feature = "cache")]
                        let g2_prep = if let Some(ref c) = cache {
                            c.g2_prepared[*orig_idx].clone()
                        } else {
                            g2_prep
                        };

                        let result = Bn254::final_exponentiation(
                            Bn254::miller_loop(g1_prep, g2_prep),
                        )
                        .expect("Final exponentiation should not fail");
                        ArkGT(result.0)
                    })
                    .collect();
                results
            })
            .collect()
    }

    /// Sequential version of multi_pair_g2_setup_batch_individual
    #[cfg(not(feature = "parallel"))]
    pub(super) fn multi_pair_g2_setup_batch_individual(ps: &[ArkG1], qs: &[ArkG2]) -> Vec<ArkGT> {
        use ark_bn254::G2Affine;

        ps.iter()
            .zip(qs.iter())
            .map(|(p, q)| {
                let g2_affine: G2Affine = q.0.into();
                let g2_prep: <Bn254 as Pairing>::G2Prepared = g2_affine.into();
                let g1_prep: <Bn254 as Pairing>::G1Prepared = p.0.into();
                let result = Bn254::final_exponentiation(
                    Bn254::miller_loop(g1_prep, g2_prep),
                )
                .expect("Final exponentiation should not fail");
                ArkGT(result.0)
            })
            .collect()
    }

    /// Batch pairing with G1 from setup, returning individual GT results.
    #[cfg(feature = "parallel")]
    pub(super) fn multi_pair_g1_setup_batch_individual(ps: &[ArkG1], qs: &[ArkG2]) -> Vec<ArkGT> {
        use rayon::prelude::*;

        let n = qs.len();
        if n == 0 {
            return vec![];
        }

        // Batch convert all G2 points to affine
        let qs_affine = batch_g2_to_affine(qs);

        let chunk_size = determine_chunk_size(n);

        #[cfg(feature = "cache")]
        let cache = crate::backends::arkworks::ark_cache::get_prepared_cache();

        qs_affine
            .par_chunks(chunk_size)
            .enumerate()
            .flat_map_iter(|(chunk_idx, qs_chunk)| {
                let start_idx = chunk_idx * chunk_size;

                qs_chunk
                    .iter()
                    .enumerate()
                    .map(|(local_idx, &q_affine)| {
                        let g2_prep: <Bn254 as Pairing>::G2Prepared = q_affine.into();

                        #[cfg(feature = "cache")]
                        let g1_prep: <Bn254 as Pairing>::G1Prepared = if let Some(ref c) = cache {
                            c.g1_prepared[start_idx + local_idx].clone()
                        } else {
                            let g1_affine: ark_bn254::G1Affine = ps[start_idx + local_idx].0.into();
                            g1_affine.into()
                        };
                        #[cfg(not(feature = "cache"))]
                        let g1_prep: <Bn254 as Pairing>::G1Prepared = {
                            let g1_affine: ark_bn254::G1Affine = ps[start_idx + local_idx].0.into();
                            g1_affine.into()
                        };

                        let result = Bn254::final_exponentiation(
                            Bn254::miller_loop(g1_prep, g2_prep),
                        )
                        .expect("Final exponentiation should not fail");
                        ArkGT(result.0)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Sequential version of multi_pair_g1_setup_batch_individual
    #[cfg(not(feature = "parallel"))]
    pub(super) fn multi_pair_g1_setup_batch_individual(ps: &[ArkG1], qs: &[ArkG2]) -> Vec<ArkGT> {
        use ark_bn254::G1Affine;

        ps.iter()
            .zip(qs.iter())
            .map(|(p, q)| {
                let g1_affine: G1Affine = p.0.into();
                let g1_prep: <Bn254 as Pairing>::G1Prepared = g1_affine.into();
                let g2_prep: <Bn254 as Pairing>::G2Prepared = q.0.into();
                let result = Bn254::final_exponentiation(
                    Bn254::miller_loop(g1_prep, g2_prep),
                )
                .expect("Final exponentiation should not fail");
                ArkGT(result.0)
            })
            .collect()
    }

    /// Compute two separate pairing products using a single Miller loop.
    ///
    /// Given pairs (a0[i], b0[i]) and (a1[i], b1[i]) for i=0..n,
    /// computes Π e(a0[i], b0[i]) and Π e(a1[i], b1[i]) with one Miller loop.
    ///
    /// Returns (product0, product1) where:
    /// - product0 = Π e(a0[i], b0[i])  (first set of pairs)
    /// - product1 = Π e(a1[i], b1[i])  (second set of pairs)
    #[cfg(feature = "parallel")]
    #[tracing::instrument(skip_all, name = "multi_pair_two_products")]
    pub(super) fn multi_pair_two_products(
        a0: &[ArkG1],
        b0: &[ArkG2],
        a1: &[ArkG1],
        b1: &[ArkG2],
    ) -> (ArkGT, ArkGT) {
        let n = a0.len();
        assert_eq!(a0.len(), b0.len());
        assert_eq!(a1.len(), b1.len());
        assert_eq!(n, a1.len());

        if n == 0 {
            return (
                ArkGT(<<Bn254 as Pairing>::TargetField>::one()),
                ArkGT(<<Bn254 as Pairing>::TargetField>::one()),
            );
        }

        let mut prod0 = <<Bn254 as Pairing>::TargetField>::one();
        let mut prod1 = <<Bn254 as Pairing>::TargetField>::one();

        for i in 0..n {
            // First pair: (a0[i], b0[i])
            let g1_a: ark_bn254::G1Affine = a0[i].0.into();
            let g2_b: ark_bn254::G2Affine = b0[i].0.into();
            let p_prep: <Bn254 as Pairing>::G1Prepared = g1_a.into();
            let q_prep: <Bn254 as Pairing>::G2Prepared = g2_b.into();
            let single_miller = Bn254::miller_loop(p_prep, q_prep);
            prod0 *= single_miller.0;

            // Second pair: (a1[i], b1[i])
            let g1_a: ark_bn254::G1Affine = a1[i].0.into();
            let g2_b: ark_bn254::G2Affine = b1[i].0.into();
            let p_prep: <Bn254 as Pairing>::G1Prepared = g1_a.into();
            let q_prep: <Bn254 as Pairing>::G2Prepared = g2_b.into();
            let single_miller = Bn254::miller_loop(p_prep, q_prep);
            prod1 *= single_miller.0;
        }

        let result0 = Bn254::final_exponentiation(ark_ec::pairing::MillerLoopOutput(prod0))
            .expect("Final exponentiation should not fail");
        let result1 = Bn254::final_exponentiation(ark_ec::pairing::MillerLoopOutput(prod1))
            .expect("Final exponentiation should not fail");

        (ArkGT(result0.0), ArkGT(result1.0))
    }

    /// Sequential version of multi_pair_two_products
    #[cfg(not(feature = "parallel"))]
    pub(super) fn multi_pair_two_products(
        a0: &[ArkG1],
        b0: &[ArkG2],
        a1: &[ArkG1],
        b1: &[ArkG2],
    ) -> (ArkGT, ArkGT) {
        use ark_bn254::G1Affine;

        let n = a0.len();
        if n == 0 {
            return (
                ArkGT(<<Bn254 as Pairing>::TargetField>::one()),
                ArkGT(<<Bn254 as Pairing>::TargetField>::one()),
            );
        }

        let mut prod0 = <<Bn254 as Pairing>::TargetField>::one();
        let mut prod1 = <<Bn254 as Pairing>::TargetField>::one();

        for i in 0..n {
            let g1_a: G1Affine = a0[i].0.into();
            let g2_b: ark_bn254::G2Affine = b0[i].0.into();
            let p_prep: <Bn254 as Pairing>::G1Prepared = g1_a.into();
            let q_prep: <Bn254 as Pairing>::G2Prepared = g2_b.into();
            let single_miller = Bn254::miller_loop(p_prep, q_prep);
            prod0 *= single_miller.0;
        }

        for i in 0..n {
            let g1_a: G1Affine = a1[i].0.into();
            let g2_b: ark_bn254::G2Affine = b1[i].0.into();
            let p_prep: <Bn254 as Pairing>::G1Prepared = g1_a.into();
            let q_prep: <Bn254 as Pairing>::G2Prepared = g2_b.into();
            let single_miller = Bn254::miller_loop(p_prep, q_prep);
            prod1 *= single_miller.0;
        }

        let result0 = Bn254::final_exponentiation(ark_ec::pairing::MillerLoopOutput(prod0))
            .expect("Final exponentiation should not fail");
        let result1 = Bn254::final_exponentiation(ark_ec::pairing::MillerLoopOutput(prod1))
            .expect("Final exponentiation should not fail");

        (ArkGT(result0.0), ArkGT(result1.0))
    }
}

impl PairingCurve for BN254 {
    type G1 = ArkG1;
    type G2 = ArkG2;
    type GT = ArkGT;

    fn pair(p: &Self::G1, q: &Self::G2) -> Self::GT {
        ArkGT(Bn254::pairing(p.0, q.0).0)
    }

    #[tracing::instrument(skip_all, name = "BN254::multi_pair", fields(len = ps.len()))]
    fn multi_pair(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair requires equal length vectors"
        );

        if ps.is_empty() {
            return Self::GT::identity();
        }

        pairing_helpers::multi_pair_optimized(ps, qs)
    }

    #[tracing::instrument(skip_all, name = "BN254::multi_pair_g2_setup", fields(len = ps.len()))]
    fn multi_pair_g2_setup(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair_g2_setup requires equal length vectors"
        );

        if ps.is_empty() {
            return Self::GT::identity();
        }

        pairing_helpers::multi_pair_g2_setup_optimized(ps, qs)
    }

    #[tracing::instrument(skip_all, name = "BN254::multi_pair_g1_setup", fields(len = ps.len()))]
    fn multi_pair_g1_setup(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair_g1_setup requires equal length vectors"
        );

        if ps.is_empty() {
            return Self::GT::identity();
        }

        pairing_helpers::multi_pair_g1_setup_optimized(ps, qs)
    }

    fn multi_pair_batch(ps: &[Self::G1], qs: &[Self::G2]) -> Vec<Self::GT> {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair_batch requires equal length vectors"
        );

        if ps.is_empty() {
            return vec![];
        }

        pairing_helpers::multi_pair_batch_individual(ps, qs)
    }

    fn multi_pair_g2_setup_batch(ps: &[Self::G1], qs: &[Self::G2]) -> Vec<Self::GT> {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair_g2_setup_batch requires equal length vectors"
        );

        if ps.is_empty() {
            return vec![];
        }

        pairing_helpers::multi_pair_g2_setup_batch_individual(ps, qs)
    }

    fn multi_pair_g1_setup_batch(ps: &[Self::G1], qs: &[Self::G2]) -> Vec<Self::GT> {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair_g1_setup_batch requires equal length vectors"
        );

        if ps.is_empty() {
            return vec![];
        }

        pairing_helpers::multi_pair_g1_setup_batch_individual(ps, qs)
    }

    fn multi_pair_two_products(
        a0: &[Self::G1],
        b0: &[Self::G2],
        a1: &[Self::G1],
        b1: &[Self::G2],
    ) -> (Self::GT, Self::GT) {
        pairing_helpers::multi_pair_two_products(a0, b0, a1, b1)
    }
}
