//! Benchmarks for lattice + GPU operations
//!
//! Run with:
//! cargo bench --bench lattice_gpu --features zkmetal

use std::time::Instant;

#[cfg(feature = "zkmetal")]
use jolt_core::poly::commitment::lattice::zkmetal_ntt::{
    cpu_ntt, ZkMetalNttEngine, ZkMetalError};
use ark_bn254::Fr;
use ark_ff::Field;

/// Benchmark GPU vs CPU for NTT of various sizes
#[cfg(feature = "zkmetal")]
fn bench_ntt_size(log_n: usize, trials: usize) -> Result<(f64, f64), ZkMetalError> {
    let n = 1 << log_n;
    let input: Vec<Fr> = (0..n).map(|i| Fr::from(i as u64)).collect();

    // GPU benchmark
    let gpu_engine = ZkMetalNttEngine::new(log_n);

    // Warmup
    for _ in 0..3 {
        let _ = gpu_engine.forward(&input)?;
    }

    let start = Instant::now();
    for _ in 0..trials {
        let _ = std::hint::black_box(gpu_engine.forward(&input)?);
    }
    let gpu_time = start.elapsed().as_secs_f64() / trials as f64;

    // CPU benchmark (naive NTT - O(n^2), so use smaller sizes or fewer trials)
    let omega = cpu_ntt::get_root_of_unity(log_n);
    let cpu_trials = if log_n <= 10 { trials } else { 1 };

    let start = Instant::now();
    for _ in 0..cpu_trials {
        let _ = std::hint::black_box(cpu_ntt::naive_ntt(&input, &omega, log_n));
    }
    let cpu_time = start.elapsed().as_secs_f64() / cpu_trials as f64;

    Ok((gpu_time, cpu_time))
}

#[cfg(feature = "zkmetal")]
#[test]
fn bench_lattice_ntt_sizes() {
    println!("\n=== Lattice NTT Benchmarks ===");
    println!("Size\tGPU (us)\tCPU (us)\tSpeedup");
    println!("-----\t--------\t--------\t-------");

    for log_n in [8, 10, 12, 14, 16].iter() {
        let result = bench_ntt_size(*log_n, 5);

        if let Ok((gpu_time, cpu_time)) = result {
            let gpu_us = gpu_time * 1e6;
            let cpu_us = cpu_time * 1e6;
            let speedup = cpu_time / gpu_time;

            println!(
                "2^{}\t{:.1}\t\t{:.1}\t\t{:.1}x",
                log_n, gpu_us, cpu_us, speedup
            );
        } else {
            println!("2^{}\tGPU failed", log_n);
        }
    }
}

#[cfg(feature = "zkmetal")]
#[test]
fn bench_lattice_ntt_roundtrip() {
    println!("\n=== Lattice NTT Roundtrip Benchmarks ===");
    println!("Size\tForward\t\tInverse\t\tTotal");
    println!("-----\t-------\t\t-------\t\t-----");

    for log_n in [12, 14, 16].iter() {
        let n = 1 << log_n;
        let input: Vec<Fr> = (0..n).map(|i| Fr::from(i as u64)).collect();
        let engine = ZkMetalNttEngine::new(log_n);

        let trials = 5;

        // Forward
        let start = Instant::now();
        for _ in 0..trials {
            let transformed = engine.forward(&input).unwrap();
            std::hint::black_box(transformed);
        }
        let forward_time = start.elapsed().as_secs_f64() / trials as f64;

        // Inverse
        let transformed = engine.forward(&input).unwrap();
        let start = Instant::now();
        for _ in 0..trials {
            let recovered = engine.inverse(&transformed).unwrap();
            std::hint::black_box(recovered);
        }
        let inverse_time = start.elapsed().as_secs_f64() / trials as f64;

        let total = forward_time + inverse_time;

        println!(
            "2^{}\t{:.1} ms\t\t{:.1} ms\t\t{:.1} ms",
            log_n,
            forward_time * 1000.0,
            inverse_time * 1000.0,
            total * 1000.0
        );
    }
}

#[cfg(feature = "zkmetal")]
#[test]
fn bench_lattice_poly_multiplication() {
    println!("\n=== Lattice Polynomial Multiplication via NTT ===");
    println!("Size\tGPU (us)\tCPU (naive us)\tSpeedup");
    println!("-----\t--------\t--------------\t-------");

    for log_n in [10, 12, 14].iter() {
        let n = 1 << log_n;
        let a: Vec<Fr> = (0..n).map(|i| Fr::from((i + 1) as u64)).collect();
        let b: Vec<Fr> = (0..n).map(|i| Fr::from((i + 2) as u64)).collect();

        let engine = ZkMetalNttEngine::new(log_n);

        // GPU: NTT(a) * NTT(b) -> INTT
        let trials = if log_n <= 12 { 10 } else { 5 };

        let start = Instant::now();
        for _ in 0..trials {
            let a_ntt = engine.forward(&a).unwrap();
            let b_ntt = engine.forward(&b).unwrap();
            let prod = engine.multiply(&a_ntt, &b_ntt).unwrap();
            let _ = engine.inverse(&prod).unwrap();
        }
        let gpu_time = start.elapsed().as_secs_f64() / trials as f64;

        // CPU: naive O(n^2) polynomial multiplication (use fewer trials)
        let cpu_trials = if log_n <= 10 { trials } else { 1 };

        let start = Instant::now();
        for _ in 0..cpu_trials {
            // Naive convolution mod (x^n - 1)
            let mut result = vec![Fr::zero(); n];
            for i in 0..n {
                for j in 0..n {
                    result[(i + j) % n] += a[i] * b[j];
                }
            }
            std::hint::black_box(result);
        }
        let cpu_time = start.elapsed().as_secs_f64() / cpu_trials as f64;

        let speedup = cpu_time / gpu_time;

        println!(
            "2^{}\t{:.1}\t\t{:.1}\t\t{:.1}x",
            log_n,
            gpu_time * 1e6,
            cpu_time * 1e6,
            speedup
        );
    }
}

#[cfg(feature = "zkmetal")]
#[test]
fn bench_conversion_overhead() {
    println!("\n=== Conversion Overhead (Fr <-> zkMetal) ===");

    for log_n in [12, 14, 16].iter() {
        let n = 1 << log_n;
        let input: Vec<Fr> = (0..n).map(|i| Fr::from(i as u64)).collect();

        let trials = 100;

        // Fr -> u64 limbs
        let start = Instant::now();
        for _ in 0..trials {
            let limbs: Vec<[u64; 4]> = input.iter().map(|f| jolt_core::poly::commitment::lattice::zkmetal_ntt::fr_to_u64_limbs(f)).collect();
            std::hint::black_box(limbs);
        }
        let to_limbs_time = start.elapsed().as_secs_f64() / trials as f64;

        // u64 limbs -> Fr
        let limbs: Vec<[u64; 4]> = input.iter().map(|f| jolt_core::poly::commitment::lattice::zkmetal_ntt::fr_to_u64_limbs(f)).collect();
        let start = Instant::now();
        for _ in 0..trials {
            let fr: Vec<Fr> = limbs.iter().map(|l| jolt_core::poly::commitment::lattice::zkmetal_ntt::u64_limbs_to_fr(l)).collect();
            std::hint::black_box(fr);
        }
        let from_limbs_time = start.elapsed().as_secs_f64() / trials as f64;

        println!(
            "2^{}\tFr->Limbs: {:.2} us\tLimbs->Fr: {:.2} us\tTotal: {:.2} us",
            log_n,
            to_limbs_time * 1e6,
            from_limbs_time * 1e6,
            (to_limbs_time + from_limbs_time) * 1e6
        );
    }
}

#[cfg(feature = "zkmetal")]
#[test]
fn bench_lattice_ntt_throughput() {
    println!("\n=== NTT Throughput (elements/sec) ===");
    println!("Size\tGPU (M/s)\tCPU (M/s)\tRatio");
    println!("-----\t--------\t--------\t-----");

    for log_n in [12, 14, 16, 18, 20].iter() {
        let n = 1 << log_n;
        let input: Vec<Fr> = (0..n).map(|i| Fr::from(i as u64)).collect();

        let engine = ZkMetalNttEngine::new(log_n);

        // Skip CPU for large sizes (too slow)
        if *log_n <= 14 {
            let trials = 10;

            // GPU
            let start = Instant::now();
            for _ in 0..trials {
                let _ = engine.forward(&input).unwrap();
            }
            let gpu_time = start.elapsed().as_secs_f64() / trials as f64;
            let gpu_throughput = n as f64 / gpu_time / 1e6;

            // CPU
            let omega = cpu_ntt::get_root_of_unity(*log_n);
            let start = Instant::now();
            for _ in 0..trials {
                let _ = cpu_ntt::naive_ntt(&input, &omega, *log_n);
            }
            let cpu_time = start.elapsed().as_secs_f64() / trials as f64;
            let cpu_throughput = n as f64 / cpu_time / 1e6;

            println!(
                "2^{}\t{:.1}\t\t{:.1}\t\t{:.1}x",
                log_n,
                gpu_throughput,
                cpu_throughput,
                gpu_throughput / cpu_throughput
            );
        } else {
            // GPU only for large sizes
            let trials = 5;
            let start = Instant::now();
            for _ in 0..trials {
                let _ = engine.forward(&input).unwrap();
            }
            let gpu_time = start.elapsed().as_secs_f64() / trials as f64;
            let gpu_throughput = n as f64 / gpu_time / 1e6;

            println!("2^{}\t{:.1}\t\tN/A\t\tN/A", log_n, gpu_throughput);
        }
    }
}

// When zkmetal feature is not enabled, provide a stub test
#[cfg(not(feature = "zkmetal"))]
#[test]
fn bench_lattice_skipped() {
    println!("Lattice GPU benchmarks skipped: zkmetal feature not enabled");
    println!("Run with: cargo test --features zkmetal --bench lattice_gpu");
}
