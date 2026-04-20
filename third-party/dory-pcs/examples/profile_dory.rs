//! Dory prover profiling utility
//!
//! Run: cargo build --release -p dory-pcs --features "backends,parallel" --example profile_dory && ./target/release/examples/profile_dory
//!
//! For GPU benchmarking with ICICLE:
//!   cargo build --release -p dory-pcs --features "backends,parallel,icicle" --example profile_dory
//!   ICICLE_CUDA_DEVICE=0 ./target/release/examples/profile_dory --gpu
//!
//! For Dory-Prime profiling (parallel reduce-and-fold):
//!   cargo build --release -p dory-pcs --features "backends,parallel,icicle,dory-prime" --example profile_dory
//!   ./target/release/examples/profile_dory --prime
//!
//! ## Size Coverage
//!
//! nu=sigma=6 (2^12 coefficients): largest MSM = 2^6 = 64 (below threshold)
//!   => CPU mode: all MSMs use arkworks
//!   => GPU mode: same as CPU (no MSMs hit 512 threshold)
//!
//! nu=sigma=9 (2^18 coefficients): largest MSM = 2^9 = 512 (at threshold)
//!   => GPU mode: VMV phase MSMs use ICICLE
//!   => Reduce-and-fold: all rounds use ICICLE for 512-element MSMs
//!
//! nu=sigma=12 (2^24 coefficients): largest MSM = 2^12 = 4096
//!   => GPU mode: ALL phases use ICICLE - best speedup (~2-4x)

use std::time::Instant;
use std::env;

fn main() {
    let use_gpu = env::args().any(|arg| arg == "--gpu");
    let use_prime = env::args().any(|arg| arg == "--prime");

    println!("Dory Profiler - Proving Time Breakdown\n");
    println!("======================================");
    println!("Mode: {} / {}\n",
        if use_gpu { "GPU (Hybrid ICICLE)" } else { "CPU (Arkworks)" },
        if use_prime { "Dory-Prime" } else { "Standard Dory" }
    );

    // Test various sizes: nu=sigma (square)
    let sizes = [(8, 4, 4), (10, 5, 5), (12, 6, 6)];

    for (num_vars, nu, sigma) in sizes {
        if let Err(e) = profile_size(num_vars, nu, sigma, use_gpu, use_prime) {
            eprintln!("Failed for 2^{}: {}", 1 << num_vars, e);
        }
    }
}

#[cfg(not(feature = "icicle"))]
fn profile_size(num_vars: usize, nu: usize, sigma: usize, _use_gpu: bool, use_prime: bool) -> Result<(), Box<dyn std::error::Error>> {
    use dory_pcs::backends::arkworks::{
        ArkFr, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254,
    };
    use dory_pcs::mode::Transparent;
    use dory_pcs::primitives::arithmetic::Field;
    use dory_pcs::primitives::poly::Polynomial;
    use dory_pcs::{prove, setup, verify};

    #[cfg(feature = "dory-prime")]
    use dory_pcs::dory_prime::prove_dory_prime;

    if _use_gpu {
        println!("  [SKIP] GPU mode requires `icicle` feature. Rebuild with: --features \"backends,parallel,icicle\"\n");
        return Ok(());
    }

    let poly_size = 1 << num_vars;
    let (prover_setup, verifier_setup) = setup::<BN254>(num_vars);

    let coefficients: Vec<ArkFr> = (0..poly_size).map(|_| ArkFr::random()).collect();
    let poly = ArkworksPolynomial::new(coefficients);

    let point: Vec<ArkFr> = (0..num_vars).map(|_| ArkFr::random()).collect();
    let evaluation = poly.evaluate(&point);

    let (tier_2, tier_1, commit_blind) = poly
        .commit::<BN254, Transparent, G1Routines>(nu, sigma, &prover_setup)?;

    // Benchmark prove
    let iterations = 5;
    let mut prove_times = Vec::with_capacity(iterations);

    if use_prime {
        // Dory-Prime profiling
        #[cfg(feature = "dory-prime")]
        {
            // Warmup
            let _ = prove_dory_prime::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
                &poly, &point, Some(tier_1.clone()), commit_blind, nu, sigma, &prover_setup,
                &mut Blake2bTranscript::new(b"profile-dory-prime"),
            );

            for _ in 0..iterations {
                let mut transcript = Blake2bTranscript::new(b"profile-dory-prime");
                let start = Instant::now();
                let proof = prove_dory_prime::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
                    &poly, &point, Some(tier_1.clone()), commit_blind, nu, sigma, &prover_setup,
                    &mut transcript,
                )?;
                prove_times.push(start.elapsed());
                drop(proof);
            }

            prove_times.sort_by_key(|d| d.as_millis());
            let prove_median = prove_times[iterations / 2];

            // Generate proof for verification
            let mut prove_transcript = Blake2bTranscript::new(b"profile-dory-prime");
            let proof = prove_dory_prime::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
                &poly, &point, Some(tier_1.clone()), commit_blind, nu, sigma, &prover_setup,
                &mut prove_transcript,
            )?;

            // Benchmark verify
            let mut transcript = Blake2bTranscript::new(b"profile-dory-prime");
            let verify_start = Instant::now();
            verify::<ArkFr, BN254, G1Routines, G2Routines, _>(
                tier_2, evaluation, &point, &proof.batch_proof, verifier_setup, &mut transcript
            )?;
            let verify_time = verify_start.elapsed();

            println!("2^{} coefficients (nu={}, sigma={}):", poly_size, nu, sigma);
            println!("  Prove median: {}ms [CPU-DORY-PRIME]", prove_median.as_millis());
            println!("  Verify:       {}ms", verify_time.as_millis());
            println!();
        }

        #[cfg(not(feature = "dory-prime"))]
        {
            println!("  [SKIP] Dory-Prime requires `dory-prime` feature. Rebuild with: --features \"backends,parallel,dory-prime\"\n");
        }
    } else {
        // Standard Dory profiling
        // Warmup
        let _ = prove::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
            &poly, &point, tier_1.clone(), commit_blind, nu, sigma, &prover_setup,
            &mut Blake2bTranscript::new(b"profile-dory"),
        );

        for _ in 0..iterations {
            let mut transcript = Blake2bTranscript::new(b"profile-dory");
            let start = Instant::now();
            let (proof, _) = prove::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
                &poly, &point, tier_1.clone(), commit_blind, nu, sigma, &prover_setup,
                &mut transcript,
            )?;
            prove_times.push(start.elapsed());
            drop(proof);
        }

        prove_times.sort_by_key(|d| d.as_millis());
        let prove_median = prove_times[iterations / 2];

        // Generate proof for verification
        let mut prove_transcript = Blake2bTranscript::new(b"profile-dory");
        let (proof, _) = prove::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
            &poly, &point, tier_1.clone(), commit_blind, nu, sigma, &prover_setup,
            &mut prove_transcript,
        )?;

        // Benchmark verify
        let mut transcript = Blake2bTranscript::new(b"profile-dory");
        let verify_start = Instant::now();
        verify::<ArkFr, BN254, G1Routines, G2Routines, _>(
            tier_2, evaluation, &point, &proof, verifier_setup, &mut transcript
        )?;
        let verify_time = verify_start.elapsed();

        println!("2^{} coefficients (nu={}, sigma={}):", poly_size, nu, sigma);
        println!("  Prove median: {}ms [CPU-ARKWORKS]", prove_median.as_millis());
        println!("  Verify:       {}ms", verify_time.as_millis());
        println!();
    }

    Ok(())
}

#[cfg(feature = "icicle")]
fn profile_size(num_vars: usize, nu: usize, sigma: usize, use_gpu: bool, use_prime: bool) -> Result<(), Box<dyn std::error::Error>> {
    use dory_pcs::backends::arkworks::{
        ArkFr, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254,
    };
    use dory_pcs::backends::hybrid::{HybridG1Routines, HybridG2Routines};
    use dory_pcs::mode::Transparent;
    use dory_pcs::primitives::arithmetic::Field;
    use dory_pcs::primitives::poly::Polynomial;
    use dory_pcs::{prove, setup, verify};

    #[cfg(feature = "dory-prime")]
    use dory_pcs::dory_prime::prove_dory_prime;

    let poly_size = 1 << num_vars;
    let (prover_setup, verifier_setup) = setup::<BN254>(num_vars);

    let coefficients: Vec<ArkFr> = (0..poly_size).map(|_| ArkFr::random()).collect();
    let poly = ArkworksPolynomial::new(coefficients);

    let point: Vec<ArkFr> = (0..num_vars).map(|_| ArkFr::random()).collect();
    let evaluation = poly.evaluate(&point);

    if use_prime {
        // Dory-Prime profiling
        #[cfg(feature = "dory-prime")]
        {
            if use_gpu {
                // Dory-Prime with Hybrid GPU routines
                let (tier_2, tier_1, commit_blind) = poly
                    .commit::<BN254, Transparent, HybridG1Routines>(nu, sigma, &prover_setup)?;

                let iterations = 5;
                let mut prove_times = Vec::with_capacity(iterations);

                // Warmup
                let _ = prove_dory_prime::<ArkFr, BN254, HybridG1Routines, HybridG2Routines, _, _, Transparent>(
                    &poly, &point, Some(tier_1.clone()), commit_blind, nu, sigma, &prover_setup,
                    &mut Blake2bTranscript::new(b"profile-dory-prime-gpu"),
                );

                for _ in 0..iterations {
                    let mut transcript = Blake2bTranscript::new(b"profile-dory-prime-gpu");
                    let start = Instant::now();
                    let proof = prove_dory_prime::<ArkFr, BN254, HybridG1Routines, HybridG2Routines, _, _, Transparent>(
                        &poly, &point, Some(tier_1.clone()), commit_blind, nu, sigma, &prover_setup,
                        &mut transcript,
                    )?;
                    prove_times.push(start.elapsed());
                    drop(proof);
                }

                prove_times.sort_by_key(|d| d.as_millis());
                let prove_median = prove_times[iterations / 2];

                // Generate proof for verification
                let mut prove_transcript = Blake2bTranscript::new(b"profile-dory-prime-gpu");
                let proof = prove_dory_prime::<ArkFr, BN254, HybridG1Routines, HybridG2Routines, _, _, Transparent>(
                    &poly, &point, Some(tier_1.clone()), commit_blind, nu, sigma, &prover_setup,
                    &mut prove_transcript,
                )?;

                let mut transcript = Blake2bTranscript::new(b"profile-dory-prime-gpu");
                let verify_start = Instant::now();
                verify::<ArkFr, BN254, HybridG1Routines, HybridG2Routines, _>(
                    tier_2, evaluation, &point, &proof.batch_proof, verifier_setup, &mut transcript
                )?;
                let verify_time = verify_start.elapsed();

                println!("2^{} coefficients (nu={}, sigma={}):", poly_size, nu, sigma);
                println!("  Prove median: {}ms [GPU-DORY-PRIME]", prove_median.as_millis());
                println!("  Verify:       {}ms", verify_time.as_millis());
                println!();
            } else {
                // Dory-Prime with CPU routines
                let (tier_2, tier_1, commit_blind) = poly
                    .commit::<BN254, Transparent, G1Routines>(nu, sigma, &prover_setup)?;

                let iterations = 5;
                let mut prove_times = Vec::with_capacity(iterations);

                // Warmup
                let _ = prove_dory_prime::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
                    &poly, &point, Some(tier_1.clone()), commit_blind, nu, sigma, &prover_setup,
                    &mut Blake2bTranscript::new(b"profile-dory-prime-cpu"),
                );

                for _ in 0..iterations {
                    let mut transcript = Blake2bTranscript::new(b"profile-dory-prime-cpu");
                    let start = Instant::now();
                    let proof = prove_dory_prime::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
                        &poly, &point, Some(tier_1.clone()), commit_blind, nu, sigma, &prover_setup,
                        &mut transcript,
                    )?;
                    prove_times.push(start.elapsed());
                    drop(proof);
                }

                prove_times.sort_by_key(|d| d.as_millis());
                let prove_median = prove_times[iterations / 2];

                // Generate proof for verification
                let mut prove_transcript = Blake2bTranscript::new(b"profile-dory-prime-cpu");
                let proof = prove_dory_prime::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
                    &poly, &point, Some(tier_1.clone()), commit_blind, nu, sigma, &prover_setup,
                    &mut prove_transcript,
                )?;

                let mut transcript = Blake2bTranscript::new(b"profile-dory-prime-cpu");
                let verify_start = Instant::now();
                verify::<ArkFr, BN254, G1Routines, G2Routines, _>(
                    tier_2, evaluation, &point, &proof.batch_proof, verifier_setup, &mut transcript
                )?;
                let verify_time = verify_start.elapsed();

                println!("2^{} coefficients (nu={}, sigma={}):", poly_size, nu, sigma);
                println!("  Prove median: {}ms [CPU-DORY-PRIME]", prove_median.as_millis());
                println!("  Verify:       {}ms", verify_time.as_millis());
                println!();
            }
        }

        #[cfg(not(feature = "dory-prime"))]
        {
            println!("  [SKIP] Dory-Prime requires `dory-prime` feature. Rebuild with: --features \"backends,parallel,icicle,dory-prime\"\n");
        }
    } else if use_gpu {
        // Standard Dory with Hybrid GPU routines
        let (tier_2, tier_1, commit_blind) = poly
            .commit::<BN254, Transparent, HybridG1Routines>(nu, sigma, &prover_setup)?;

        let iterations = 5;
        let mut prove_times = Vec::with_capacity(iterations);

        // Warmup
        let _ = prove::<ArkFr, BN254, HybridG1Routines, HybridG2Routines, _, _, Transparent>(
            &poly, &point, tier_1.clone(), commit_blind, nu, sigma, &prover_setup,
            &mut Blake2bTranscript::new(b"profile-dory-gpu"),
        );

        for _ in 0..iterations {
            let mut transcript = Blake2bTranscript::new(b"profile-dory-gpu");
            let start = Instant::now();
            let (proof, _) = prove::<ArkFr, BN254, HybridG1Routines, HybridG2Routines, _, _, Transparent>(
                &poly, &point, tier_1.clone(), commit_blind, nu, sigma, &prover_setup,
                &mut transcript,
            )?;
            prove_times.push(start.elapsed());
            drop(proof);
        }

        prove_times.sort_by_key(|d| d.as_millis());
        let prove_median = prove_times[iterations / 2];

        // Generate proof for verification
        let mut prove_transcript = Blake2bTranscript::new(b"profile-dory-gpu");
        let (proof, _) = prove::<ArkFr, BN254, HybridG1Routines, HybridG2Routines, _, _, Transparent>(
            &poly, &point, tier_1.clone(), commit_blind, nu, sigma, &prover_setup,
            &mut prove_transcript,
        )?;

        let mut transcript = Blake2bTranscript::new(b"profile-dory-gpu");
        let verify_start = Instant::now();
        verify::<ArkFr, BN254, HybridG1Routines, HybridG2Routines, _>(
            tier_2, evaluation, &point, &proof, verifier_setup, &mut transcript
        )?;
        let verify_time = verify_start.elapsed();

        println!("2^{} coefficients (nu={}, sigma={}):", poly_size, nu, sigma);
        println!("  Prove median: {}ms [GPU-HYBRID]", prove_median.as_millis());
        println!("  Verify:       {}ms", verify_time.as_millis());
        println!();

        Ok(())
    } else {
        // CPU-only mode: use arkworks routines
        let (tier_2, tier_1, commit_blind) = poly
            .commit::<BN254, Transparent, G1Routines>(nu, sigma, &prover_setup)?;

        let iterations = 5;
        let mut prove_times = Vec::with_capacity(iterations);

        // Warmup
        let _ = prove::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
            &poly, &point, tier_1.clone(), commit_blind, nu, sigma, &prover_setup,
            &mut Blake2bTranscript::new(b"profile-dory-cpu"),
        );

        for _ in 0..iterations {
            let mut transcript = Blake2bTranscript::new(b"profile-dory-cpu");
            let start = Instant::now();
            let (proof, _) = prove::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
                &poly, &point, tier_1.clone(), commit_blind, nu, sigma, &prover_setup,
                &mut transcript,
            )?;
            prove_times.push(start.elapsed());
            drop(proof);
        }

        prove_times.sort_by_key(|d| d.as_millis());
        let prove_median = prove_times[iterations / 2];

        // Generate proof for verification
        let mut prove_transcript = Blake2bTranscript::new(b"profile-dory-cpu");
        let (proof, _) = prove::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
            &poly, &point, tier_1.clone(), commit_blind, nu, sigma, &prover_setup,
            &mut prove_transcript,
        )?;

        let mut transcript = Blake2bTranscript::new(b"profile-dory-cpu");
        let verify_start = Instant::now();
        verify::<ArkFr, BN254, G1Routines, G2Routines, _>(
            tier_2, evaluation, &point, &proof, verifier_setup, &mut transcript
        )?;
        let verify_time = verify_start.elapsed();

        println!("2^{} coefficients (nu={}, sigma={}):", poly_size, nu, sigma);
        println!("  Prove median: {}ms [CPU-ARKWORKS]", prove_median.as_millis());
        println!("  Verify:       {}ms", verify_time.as_millis());
        println!();

        Ok(())
    }
}
