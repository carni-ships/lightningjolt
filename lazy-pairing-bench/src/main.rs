//! Benchmark: Constantine vs Arkworks BN254 pairing
//!
//! Compares single pairing, multi-miller loop, and full multi-pairing performance.

use ark_bn254::{Bn254, Fq, Fq12, Fq2, G1Affine, G2Affine};
use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;
use ark_std::test_rng;
use rayon::prelude::*;
use std::time::Instant;

// ============================================================================
// Constantine FFI
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone)]
struct CttFp12([u8; 384]);

#[repr(C)]
#[derive(Copy, Clone)]
struct CttG1Aff([u8; 64]);

#[repr(C)]
#[derive(Copy, Clone)]
struct CttG2Aff([u8; 128]);

unsafe extern "C" {
    fn ctt_bn254_pairing(gt: *mut CttFp12, p: *const CttG1Aff, q: *const CttG2Aff);
    fn ctt_bn254_multi_pairing(
        gt: *mut CttFp12,
        ps: *const CttG1Aff,
        qs: *const CttG2Aff,
        n: usize,
    );
    fn ctt_bn254_miller_loop(f: *mut CttFp12, q: *const CttG2Aff, p: *const CttG1Aff);
    fn ctt_bn254_multi_miller_loop(
        f: *mut CttFp12,
        qs: *const CttG2Aff,
        ps: *const CttG1Aff,
        n: usize,
    );
    fn ctt_bn254_final_exp(f: *mut CttFp12);
    fn ctt_bn254_fp12_mul(r: *mut CttFp12, a: *const CttFp12, b: *const CttFp12);
    fn ctt_bn254_fp12_set_one(r: *mut CttFp12);
}

/// Convert arkworks G1Affine to Constantine layout.
/// Both use Montgomery form for Fp with same modulus.
/// Layout: [x: Fp(4×u64), y: Fp(4×u64)] = 64 bytes
fn g1_to_ctt(p: &G1Affine) -> CttG1Aff {
    unsafe { std::mem::transmute_copy(p) }
}

/// Convert arkworks G2Affine to Constantine layout.
/// Layout: [x: Fp2(2×Fp), y: Fp2(2×Fp)] = 128 bytes
fn g2_to_ctt(q: &G2Affine) -> CttG2Aff {
    unsafe { std::mem::transmute_copy(q) }
}

/// Convert Constantine Fp12 result back to arkworks for comparison.
/// Tower coordinate ordering differs — permutation discovered empirically:
/// ctt[0,1,2,3,4,5,6,7,8,9,10,11] → ark[0,1,8,9,6,7,4,5,2,3,10,11]
fn ctt_fp12_to_ark(f: &CttFp12) -> Fq12 {
    let ctt = unsafe { &*(f as *const CttFp12 as *const [[u64; 4]; 12]) };
    let mut ark = [[0u64; 4]; 12];
    // Permutation: ctt index → ark index
    let perm = [0, 1, 8, 9, 6, 7, 4, 5, 2, 3, 10, 11];
    for i in 0..12 {
        ark[perm[i]] = ctt[i];
    }
    unsafe { std::mem::transmute(ark) }
}

// ============================================================================
// Benchmark helpers
// ============================================================================

fn bench<F: FnMut()>(name: &str, iters: u64, mut f: F) -> f64 {
    // Warmup
    for _ in 0..iters / 10 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() as f64 / iters as f64;
    let us_per_op = ns_per_op / 1000.0;
    println!(
        "  {:<45} {:>8.1} ns/op  ({:>8.2} µs/op)  [{} iters, {:?}]",
        name, ns_per_op, us_per_op, iters, elapsed
    );
    ns_per_op
}

fn main() {
    let mut rng = test_rng();

    // Generate test points
    let g1 = G1Affine::rand(&mut rng);
    let g2 = G2Affine::rand(&mut rng);

    // ========================================================================
    // Correctness: verify Constantine matches arkworks
    // ========================================================================
    println!("=== Correctness verification ===\n");

    // Single pairing
    let ark_result = Bn254::pairing(g1, g2).0;
    let mut ctt_result = CttFp12([0u8; 384]);
    let ctt_g1 = g1_to_ctt(&g1);
    let ctt_g2 = g2_to_ctt(&g2);
    unsafe { ctt_bn254_pairing(&mut ctt_result, &ctt_g1, &ctt_g2) };
    let ctt_as_ark = ctt_fp12_to_ark(&ctt_result);

    println!(
        "  Single pairing match: {}",
        ark_result == ctt_as_ark
    );

    if ark_result != ctt_as_ark {
        println!("  Layout differs — dumping all 12 Fq components:");
        let ark_limbs = unsafe { &*(&ark_result as *const Fq12 as *const [[u64; 4]; 12]) };
        let ctt_limbs = unsafe { &*(&ctt_result as *const CttFp12 as *const [[u64; 4]; 12]) };
        for i in 0..12 {
            let matched = ark_limbs[i] == ctt_limbs[i];
            if !matched {
                println!("  [{}] ark={:?}", i, ark_limbs[i]);
                println!("  [{}] ctt={:?} DIFFER", i, ctt_limbs[i]);
            }
        }
        // Check if ctt components appear in ark but at different positions
        println!("\n  Looking for permutation mapping ctt->ark:");
        for ci in 0..12 {
            for ai in 0..12 {
                if ctt_limbs[ci] == ark_limbs[ai] && ci != ai {
                    println!("    ctt[{}] == ark[{}]", ci, ai);
                }
            }
        }
        // Continue benchmarking anyway — we can fix mapping later
        println!();
    }

    // Multi-pairing (n=2)
    let g1b = G1Affine::rand(&mut rng);
    let g2b = G2Affine::rand(&mut rng);
    let ark_multi = Bn254::multi_pairing([g1, g1b], [g2, g2b]).0;

    let ctt_g1s = [g1_to_ctt(&g1), g1_to_ctt(&g1b)];
    let ctt_g2s = [g2_to_ctt(&g2), g2_to_ctt(&g2b)];
    let mut ctt_multi = CttFp12([0u8; 384]);
    unsafe {
        ctt_bn254_multi_pairing(&mut ctt_multi, ctt_g1s.as_ptr(), ctt_g2s.as_ptr(), 2);
    }
    let ctt_multi_ark = ctt_fp12_to_ark(&ctt_multi);
    println!("  Multi-pairing (n=2) match: {}", ark_multi == ctt_multi_ark);

    // Miller loop + final exp
    let ark_ml = Bn254::multi_miller_loop([g1], [g2]);
    let mut ctt_ml = CttFp12([0u8; 384]);
    unsafe { ctt_bn254_miller_loop(&mut ctt_ml, &ctt_g2, &ctt_g1) };
    let mut ctt_ml_fe = ctt_ml;
    unsafe { ctt_bn254_final_exp(&mut ctt_ml_fe) };
    let ctt_ml_ark = ctt_fp12_to_ark(&ctt_ml_fe);
    println!(
        "  Miller+FinalExp match: {}",
        Bn254::final_exponentiation(ark_ml).unwrap().0 == ctt_ml_ark
    );

    // ========================================================================
    // Benchmarks
    // ========================================================================
    println!("\n=== Single pairing benchmark ===\n");

    let iters = 1000;

    let ark_ns = bench("arkworks: Bn254::pairing()", iters, || {
        std::hint::black_box(Bn254::pairing(
            std::hint::black_box(g1),
            std::hint::black_box(g2),
        ));
    });

    let ctt_ns = bench("constantine: ctt_bn254_pairing()", iters, || {
        let mut r = CttFp12([0u8; 384]);
        unsafe {
            ctt_bn254_pairing(&mut r, &ctt_g1, &ctt_g2);
        }
        std::hint::black_box(r);
    });

    println!("\n  Speedup: {:.2}x", ark_ns / ctt_ns);

    // ========================================================================
    println!("\n=== Miller loop benchmark ===\n");

    bench("arkworks: multi_miller_loop(1)", iters, || {
        std::hint::black_box(Bn254::multi_miller_loop([g1], [g2]));
    });

    bench("constantine: miller_loop(1)", iters, || {
        let mut r = CttFp12([0u8; 384]);
        unsafe { ctt_bn254_miller_loop(&mut r, &ctt_g2, &ctt_g1) };
        std::hint::black_box(r);
    });

    // ========================================================================
    println!("\n=== Final exponentiation benchmark ===\n");

    let ml_result = Bn254::multi_miller_loop([g1], [g2]);
    let ml_fq12 = ml_result.0;
    let mut ctt_ml_input = CttFp12([0u8; 384]);
    unsafe { ctt_bn254_miller_loop(&mut ctt_ml_input, &ctt_g2, &ctt_g1) };

    bench("arkworks: final_exponentiation()", iters, || {
        let mut f = ml_fq12;
        use ark_ff::fields::models::fp12_2over3over2::Fp12;
        // arkworks final_exp is on MillerLoopOutput
        let mlo = ark_ec::pairing::MillerLoopOutput::<Bn254>(f);
        std::hint::black_box(Bn254::final_exponentiation(mlo));
    });

    bench("constantine: final_exp()", iters, || {
        let mut f = ctt_ml_input;
        unsafe { ctt_bn254_final_exp(&mut f) };
        std::hint::black_box(f);
    });

    // ========================================================================
    println!("\n=== Multi-pairing benchmark ===\n");

    for n in [2, 4, 8, 16, 32, 64] {
        let g1s: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
        let g2s: Vec<G2Affine> = (0..n).map(|_| G2Affine::rand(&mut rng)).collect();
        let ctt_g1s: Vec<CttG1Aff> = g1s.iter().map(g1_to_ctt).collect();
        let ctt_g2s: Vec<CttG2Aff> = g2s.iter().map(g2_to_ctt).collect();

        let iters_n = 500 / n as u64 + 1;

        let ark_ns_n = bench(
            &format!("arkworks: multi_pairing(n={})", n),
            iters_n,
            || {
                std::hint::black_box(Bn254::multi_pairing(&g1s, &g2s));
            },
        );

        let ctt_ns_n = bench(
            &format!("constantine: multi_pairing(n={})", n),
            iters_n,
            || {
                let mut r = CttFp12([0u8; 384]);
                unsafe {
                    ctt_bn254_multi_pairing(
                        &mut r,
                        ctt_g1s.as_ptr(),
                        ctt_g2s.as_ptr(),
                        n,
                    );
                }
                std::hint::black_box(r);
            },
        );

        println!("  Speedup (n={}): {:.2}x\n", n, ark_ns_n / ctt_ns_n);
    }

    // ========================================================================
    // Simulate Dory tier2: N separate multi_miller_loops each with ~128 pairs,
    // followed by separate final exponentiations
    // ========================================================================
    println!("=== Dory-like tier2 simulation (separate pairings) ===\n");

    let batch_size = 128;
    let num_polys = 42; // from trace: 42 tier2 calls
    let g1_batches: Vec<Vec<G1Affine>> = (0..num_polys)
        .map(|_| (0..batch_size).map(|_| G1Affine::rand(&mut rng)).collect())
        .collect();
    let g2_batch: Vec<G2Affine> = (0..batch_size)
        .map(|_| G2Affine::rand(&mut rng))
        .collect();
    let ctt_g1_batches: Vec<Vec<CttG1Aff>> = g1_batches
        .iter()
        .map(|b| b.iter().map(g1_to_ctt).collect())
        .collect();
    let ctt_g2_batch: Vec<CttG2Aff> = g2_batch.iter().map(g2_to_ctt).collect();

    let tier2_iters = 5;

    bench(
        &format!(
            "arkworks: {}x multi_pairing(n={})",
            num_polys, batch_size
        ),
        tier2_iters,
        || {
            for g1_batch in &g1_batches {
                std::hint::black_box(Bn254::multi_pairing(g1_batch, &g2_batch));
            }
        },
    );

    bench(
        &format!(
            "constantine: {}x multi_pairing(n={})",
            num_polys, batch_size
        ),
        tier2_iters,
        || {
            for ctt_g1_batch in &ctt_g1_batches {
                let mut r = CttFp12([0u8; 384]);
                unsafe {
                    ctt_bn254_multi_pairing(
                        &mut r,
                        ctt_g1_batch.as_ptr(),
                        ctt_g2_batch.as_ptr(),
                        batch_size,
                    );
                }
                std::hint::black_box(r);
            }
        },
    );

    // ========================================================================
    // Parallel versions
    // ========================================================================
    println!("\n=== Dory-like tier2 simulation (PARALLEL) ===\n");

    bench(
        &format!(
            "arkworks PAR: {}x multi_pairing(n={})",
            num_polys, batch_size
        ),
        tier2_iters,
        || {
            g1_batches.par_iter().for_each(|g1_batch| {
                std::hint::black_box(Bn254::multi_pairing(g1_batch, &g2_batch));
            });
        },
    );

    bench(
        &format!(
            "constantine PAR: {}x multi_pairing(n={})",
            num_polys, batch_size
        ),
        tier2_iters,
        || {
            ctt_g1_batches.par_iter().for_each(|ctt_g1_batch| {
                let mut r = CttFp12([0u8; 384]);
                unsafe {
                    ctt_bn254_multi_pairing(
                        &mut r,
                        ctt_g1_batch.as_ptr(),
                        ctt_g2_batch.as_ptr(),
                        batch_size,
                    );
                }
                std::hint::black_box(r);
            });
        },
    );
}
