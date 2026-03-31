use super::gpu;
use ark_bn254::{Fq, Fq12, Fq2, Fq6};
use ark_ff::{Field, UniformRand};
use std::time::Instant;

#[allow(unused_imports)]
use ark_ec::bn::BnConfig;

/// Convert an arkworks Fq (Montgomery form) to [u32; 8] limbs.
fn fq_to_limbs(f: &Fq) -> [u32; 8] {
    let mont_limbs: [u64; 4] = f.0 .0;
    let mut result = [0u32; 8];
    for i in 0..4 {
        result[2 * i] = mont_limbs[i] as u32;
        result[2 * i + 1] = (mont_limbs[i] >> 32) as u32;
    }
    result
}

/// Convert [u32; 8] limbs (Montgomery form) back to arkworks Fq.
fn limbs_to_fq(limbs: &[u32; 8]) -> Fq {
    let mut u64_limbs = [0u64; 4];
    for i in 0..4 {
        u64_limbs[i] = limbs[2 * i] as u64 | ((limbs[2 * i + 1] as u64) << 32);
    }
    Fq::new_unchecked(ark_ff::BigInt(u64_limbs))
}

/// Convert arkworks Fq2 to [u32; 16] limbs (c0 then c1).
fn fq2_to_limbs(f: &Fq2) -> gpu::Fq2Limbs {
    let mut result = [0u32; 16];
    let c0 = fq_to_limbs(&f.c0);
    let c1 = fq_to_limbs(&f.c1);
    result[..8].copy_from_slice(&c0);
    result[8..].copy_from_slice(&c1);
    result
}

/// Convert [u32; 16] limbs back to arkworks Fq2.
fn limbs_to_fq2(limbs: &gpu::Fq2Limbs) -> Fq2 {
    let c0 = limbs_to_fq(&limbs[..8].try_into().unwrap());
    let c1 = limbs_to_fq(&limbs[8..].try_into().unwrap());
    Fq2::new(c0, c1)
}

/// Convert arkworks Fq6 to 48 × u32 limbs (c0.c0, c0.c1, c1.c0, c1.c1, c2.c0, c2.c1).
fn fq6_to_limbs(f: &Fq6) -> [u32; 48] {
    let mut result = [0u32; 48];
    let parts = [
        fq_to_limbs(&f.c0.c0),
        fq_to_limbs(&f.c0.c1),
        fq_to_limbs(&f.c1.c0),
        fq_to_limbs(&f.c1.c1),
        fq_to_limbs(&f.c2.c0),
        fq_to_limbs(&f.c2.c1),
    ];
    for (i, p) in parts.iter().enumerate() {
        result[i * 8..(i + 1) * 8].copy_from_slice(p);
    }
    result
}

/// Convert arkworks Fq12 to [u32; 96] limbs.
fn fq12_to_limbs(f: &Fq12) -> gpu::Fq12Limbs {
    let mut result = [0u32; 96];
    let c0 = fq6_to_limbs(&f.c0);
    let c1 = fq6_to_limbs(&f.c1);
    result[..48].copy_from_slice(&c0);
    result[48..].copy_from_slice(&c1);
    result
}

/// Convert [u32; 96] limbs back to arkworks Fq12.
fn limbs_to_fq12(limbs: &gpu::Fq12Limbs) -> Fq12 {
    let mut fq_elems = [Fq::default(); 12];
    for i in 0..12 {
        let chunk: [u32; 8] = limbs[i * 8..(i + 1) * 8].try_into().unwrap();
        fq_elems[i] = limbs_to_fq(&chunk);
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

#[test]
fn test_fq_roundtrip() {
    let mut rng = ark_std::test_rng();
    for _ in 0..100 {
        let f = Fq::rand(&mut rng);
        let limbs = fq_to_limbs(&f);
        let recovered = limbs_to_fq(&limbs);
        assert_eq!(f, recovered, "Fq roundtrip failed");
    }
}

#[test]
fn test_gpu_fq_mul_correctness() {
    let mut rng = ark_std::test_rng();
    let n = 1024;

    let a_fq: Vec<Fq> = (0..n).map(|_| Fq::rand(&mut rng)).collect();
    let b_fq: Vec<Fq> = (0..n).map(|_| Fq::rand(&mut rng)).collect();

    let expected: Vec<Fq> = a_fq.iter().zip(&b_fq).map(|(a, b)| *a * *b).collect();

    let a_limbs: Vec<[u32; 8]> = a_fq.iter().map(fq_to_limbs).collect();
    let b_limbs: Vec<[u32; 8]> = b_fq.iter().map(fq_to_limbs).collect();

    let gpu_results = gpu::gpu_fq_mul_batch(&a_limbs, &b_limbs);

    for i in 0..n {
        let gpu_fq = limbs_to_fq(&gpu_results[i]);
        assert_eq!(
            expected[i], gpu_fq,
            "Mismatch at index {i}: expected {:?}, got {:?}",
            expected[i], gpu_fq
        );
    }
    println!("GPU Fq multiply: all {n} results match arkworks");
}

#[test]
fn test_gpu_fq_mul_chain_correctness() {
    let mut rng = ark_std::test_rng();
    let n = 64;
    let iters = 100u32;

    let a_fq: Vec<Fq> = (0..n).map(|_| Fq::rand(&mut rng)).collect();
    let b_fq: Vec<Fq> = (0..n).map(|_| Fq::rand(&mut rng)).collect();

    let expected: Vec<Fq> = a_fq
        .iter()
        .zip(&b_fq)
        .map(|(a, b)| {
            let mut val = *a * *b;
            for _ in 0..iters {
                val = val.square();
            }
            val
        })
        .collect();

    let a_limbs: Vec<[u32; 8]> = a_fq.iter().map(fq_to_limbs).collect();
    let b_limbs: Vec<[u32; 8]> = b_fq.iter().map(fq_to_limbs).collect();

    let gpu_results = gpu::gpu_fq_mul_chain(&a_limbs, &b_limbs, iters);

    for i in 0..n {
        let gpu_fq = limbs_to_fq(&gpu_results[i]);
        assert_eq!(
            expected[i], gpu_fq,
            "Chain mismatch at index {i} after {iters} squarings"
        );
    }
    println!("GPU Fq mul+chain: all {n} results match arkworks after {iters} squarings");
}

#[test]
fn test_gpu_fq2_mul_correctness() {
    let mut rng = ark_std::test_rng();
    let n = 512;

    let a_fq2: Vec<Fq2> = (0..n).map(|_| Fq2::rand(&mut rng)).collect();
    let b_fq2: Vec<Fq2> = (0..n).map(|_| Fq2::rand(&mut rng)).collect();

    let expected: Vec<Fq2> = a_fq2.iter().zip(&b_fq2).map(|(a, b)| *a * *b).collect();

    let a_limbs: Vec<gpu::Fq2Limbs> = a_fq2.iter().map(fq2_to_limbs).collect();
    let b_limbs: Vec<gpu::Fq2Limbs> = b_fq2.iter().map(fq2_to_limbs).collect();

    let gpu_results = gpu::gpu_fq2_mul_batch(&a_limbs, &b_limbs);

    for i in 0..n {
        let gpu_fq2 = limbs_to_fq2(&gpu_results[i]);
        assert_eq!(
            expected[i], gpu_fq2,
            "Fq2 mismatch at index {i}: expected {:?}, got {:?}",
            expected[i], gpu_fq2
        );
    }
    println!("GPU Fq2 multiply: all {n} results match arkworks");
}

#[test]
fn test_gpu_fq12_mul_correctness() {
    let mut rng = ark_std::test_rng();
    let n = 64;

    let a_fq12: Vec<Fq12> = (0..n).map(|_| Fq12::rand(&mut rng)).collect();
    let b_fq12: Vec<Fq12> = (0..n).map(|_| Fq12::rand(&mut rng)).collect();

    let expected: Vec<Fq12> = a_fq12.iter().zip(&b_fq12).map(|(a, b)| *a * *b).collect();

    let a_limbs: Vec<gpu::Fq12Limbs> = a_fq12.iter().map(fq12_to_limbs).collect();
    let b_limbs: Vec<gpu::Fq12Limbs> = b_fq12.iter().map(fq12_to_limbs).collect();

    let gpu_results = gpu::gpu_fq12_mul_batch(&a_limbs, &b_limbs);

    for i in 0..n {
        let gpu_fq12 = limbs_to_fq12(&gpu_results[i]);
        assert_eq!(expected[i], gpu_fq12, "Fq12 mismatch at index {i}");
    }
    println!("GPU Fq12 multiply: all {n} results match arkworks");
}

#[test]
fn test_gpu_fq12_sqr_chain_correctness() {
    let mut rng = ark_std::test_rng();
    let n = 32;
    let iters = 50u32;

    let inputs: Vec<Fq12> = (0..n).map(|_| Fq12::rand(&mut rng)).collect();

    let expected: Vec<Fq12> = inputs
        .iter()
        .map(|f| {
            let mut val = *f;
            for _ in 0..iters {
                val = val.square();
            }
            val
        })
        .collect();

    let input_limbs: Vec<gpu::Fq12Limbs> = inputs.iter().map(fq12_to_limbs).collect();

    let gpu_results = gpu::gpu_fq12_sqr_chain(&input_limbs, iters);

    for i in 0..n {
        let gpu_fq12 = limbs_to_fq12(&gpu_results[i]);
        assert_eq!(
            expected[i], gpu_fq12,
            "Fq12 sqr chain mismatch at index {i} after {iters} squarings"
        );
    }
    println!("GPU Fq12 sqr chain: all {n} results match arkworks after {iters} squarings");
}

#[test]
fn bench_gpu_vs_cpu_fq_mul() {
    let mut rng = ark_std::test_rng();
    let n = 1024;
    let warmup = 3;
    let trials = 10;

    let a_fq: Vec<Fq> = (0..n).map(|_| Fq::rand(&mut rng)).collect();
    let b_fq: Vec<Fq> = (0..n).map(|_| Fq::rand(&mut rng)).collect();
    let a_limbs: Vec<[u32; 8]> = a_fq.iter().map(fq_to_limbs).collect();
    let b_limbs: Vec<[u32; 8]> = b_fq.iter().map(fq_to_limbs).collect();

    for _ in 0..warmup {
        let _ = gpu::gpu_fq_mul_batch(&a_limbs, &b_limbs);
    }

    let start = Instant::now();
    for _ in 0..trials {
        let _ = std::hint::black_box(gpu::gpu_fq_mul_batch(&a_limbs, &b_limbs));
    }
    let gpu_time = start.elapsed() / trials;

    let start = Instant::now();
    for _ in 0..trials {
        let _: Vec<Fq> =
            std::hint::black_box(a_fq.iter().zip(&b_fq).map(|(a, b)| *a * *b).collect());
    }
    let cpu_time = start.elapsed() / trials;

    println!("\n=== Fq Multiply Benchmark ({n} elements) ===");
    println!("CPU (arkworks): {cpu_time:?}");
    println!("GPU (Metal):    {gpu_time:?}");
    let ratio = cpu_time.as_nanos() as f64 / gpu_time.as_nanos() as f64;
    println!("Ratio:          {ratio:.2}x");
}

#[test]
fn bench_gpu_fq_chain_throughput() {
    let mut rng = ark_std::test_rng();
    let n = 1024;
    let iters = 1000u32;
    let warmup = 2;
    let trials = 5;

    let a_fq: Vec<Fq> = (0..n).map(|_| Fq::rand(&mut rng)).collect();
    let b_fq: Vec<Fq> = (0..n).map(|_| Fq::rand(&mut rng)).collect();
    let a_limbs: Vec<[u32; 8]> = a_fq.iter().map(fq_to_limbs).collect();
    let b_limbs: Vec<[u32; 8]> = b_fq.iter().map(fq_to_limbs).collect();

    for _ in 0..warmup {
        let _ = gpu::gpu_fq_mul_chain(&a_limbs, &b_limbs, iters);
    }

    let start = Instant::now();
    for _ in 0..trials {
        let _ = std::hint::black_box(gpu::gpu_fq_mul_chain(&a_limbs, &b_limbs, iters));
    }
    let gpu_time = start.elapsed() / trials;

    let start = Instant::now();
    for _ in 0..trials {
        let _: Vec<Fq> = std::hint::black_box(
            a_fq.iter()
                .zip(&b_fq)
                .map(|(a, b)| {
                    let mut val = *a * *b;
                    for _ in 0..iters {
                        val = val.square();
                    }
                    val
                })
                .collect(),
        );
    }
    let cpu_time = start.elapsed() / trials;

    let total_muls = n as u64 * (iters as u64 + 1);
    let gpu_muls_per_sec = total_muls as f64 / gpu_time.as_secs_f64();
    let cpu_muls_per_sec = total_muls as f64 / cpu_time.as_secs_f64();

    println!("\n=== Fq Chain Throughput ({n} threads × {iters} squarings) ===");
    let cpu_rate = cpu_muls_per_sec / 1e6;
    let gpu_rate = gpu_muls_per_sec / 1e6;
    let ratio = cpu_time.as_nanos() as f64 / gpu_time.as_nanos() as f64;
    println!("CPU (arkworks): {cpu_time:?} ({cpu_rate:.1}M muls/sec)");
    println!("GPU (Metal):    {gpu_time:?} ({gpu_rate:.1}M muls/sec)");
    println!("Ratio:          {ratio:.2}x");
}

#[test]
fn bench_gpu_miller_loop_scale() {
    let mut rng = ark_std::test_rng();
    let n = 1024;
    let iters = 2000u32;
    let trials = 5;

    let a_fq: Vec<Fq> = (0..n).map(|_| Fq::rand(&mut rng)).collect();
    let b_fq: Vec<Fq> = (0..n).map(|_| Fq::rand(&mut rng)).collect();
    let a_limbs: Vec<[u32; 8]> = a_fq.iter().map(fq_to_limbs).collect();
    let b_limbs: Vec<[u32; 8]> = b_fq.iter().map(fq_to_limbs).collect();

    let _ = gpu::gpu_fq_mul_chain(&a_limbs, &b_limbs, iters);

    let start = Instant::now();
    for _ in 0..trials {
        let _ = std::hint::black_box(gpu::gpu_fq_mul_chain(&a_limbs, &b_limbs, iters));
    }
    let gpu_time = start.elapsed() / trials;

    let start = Instant::now();
    for _ in 0..trials {
        let _: Vec<Fq> = std::hint::black_box(
            a_fq.iter()
                .zip(&b_fq)
                .map(|(a, b)| {
                    let mut val = *a * *b;
                    for _ in 0..iters {
                        val = val.square();
                    }
                    val
                })
                .collect(),
        );
    }
    let cpu_single = start.elapsed() / trials;

    use rayon::prelude::*;
    let start = Instant::now();
    for _ in 0..trials {
        let _: Vec<Fq> = std::hint::black_box(
            a_fq.par_iter()
                .zip(&b_fq)
                .map(|(a, b)| {
                    let mut val = *a * *b;
                    for _ in 0..iters {
                        val = val.square();
                    }
                    val
                })
                .collect(),
        );
    }
    let cpu_parallel = start.elapsed() / trials;

    println!("\n=== Miller Loop Scale ({n} threads × {iters} muls) ===");
    println!("CPU single-threaded: {cpu_single:?}");
    println!("CPU parallel (rayon): {cpu_parallel:?}");
    println!("GPU (Metal):          {gpu_time:?}");
    let ratio_st = cpu_single.as_nanos() as f64 / gpu_time.as_nanos() as f64;
    let ratio_mt = cpu_parallel.as_nanos() as f64 / gpu_time.as_nanos() as f64;
    println!("vs single-thread:     {ratio_st:.1}x");
    println!("vs parallel:          {ratio_mt:.1}x");
}

#[test]
fn bench_gpu_fq12_throughput() {
    let mut rng = ark_std::test_rng();
    let n = 256;
    let iters = 100u32;
    let trials = 5;

    let inputs: Vec<Fq12> = (0..n).map(|_| Fq12::rand(&mut rng)).collect();
    let input_limbs: Vec<gpu::Fq12Limbs> = inputs.iter().map(fq12_to_limbs).collect();

    // Warmup
    let _ = gpu::gpu_fq12_sqr_chain(&input_limbs, iters);

    // GPU benchmark
    let start = Instant::now();
    for _ in 0..trials {
        let _ = std::hint::black_box(gpu::gpu_fq12_sqr_chain(&input_limbs, iters));
    }
    let gpu_time = start.elapsed() / trials;

    // CPU single-threaded benchmark
    let start = Instant::now();
    for _ in 0..trials {
        let _: Vec<Fq12> = std::hint::black_box(
            inputs
                .iter()
                .map(|f| {
                    let mut val = *f;
                    for _ in 0..iters {
                        val = val.square();
                    }
                    val
                })
                .collect(),
        );
    }
    let cpu_time = start.elapsed() / trials;

    // CPU parallel benchmark
    use rayon::prelude::*;
    let start = Instant::now();
    for _ in 0..trials {
        let _: Vec<Fq12> = std::hint::black_box(
            inputs
                .par_iter()
                .map(|f| {
                    let mut val = *f;
                    for _ in 0..iters {
                        val = val.square();
                    }
                    val
                })
                .collect(),
        );
    }
    let cpu_parallel = start.elapsed() / trials;

    let total_ops = n as u64 * iters as u64;
    let gpu_rate = total_ops as f64 / gpu_time.as_secs_f64() / 1e6;
    let cpu_rate = total_ops as f64 / cpu_time.as_secs_f64() / 1e6;

    println!("\n=== Fq12 Squaring Throughput ({n} threads × {iters} squarings) ===");
    println!("CPU single-threaded: {cpu_time:?} ({cpu_rate:.2}M sqr/sec)");
    println!("CPU parallel (rayon): {cpu_parallel:?}");
    println!("GPU (Metal):          {gpu_time:?} ({gpu_rate:.2}M sqr/sec)");
    let ratio_st = cpu_time.as_nanos() as f64 / gpu_time.as_nanos() as f64;
    let ratio_mt = cpu_parallel.as_nanos() as f64 / gpu_time.as_nanos() as f64;
    println!("vs single-thread:     {ratio_st:.1}x");
    println!("vs parallel:          {ratio_mt:.1}x");
}

#[test]
fn bench_gpu_fq12_miller_scale() {
    let mut rng = ark_std::test_rng();
    let n = 1024;
    let iters = 65u32;
    let trials = 5;

    let inputs: Vec<Fq12> = (0..n).map(|_| Fq12::rand(&mut rng)).collect();
    let input_limbs: Vec<gpu::Fq12Limbs> = inputs.iter().map(fq12_to_limbs).collect();

    let _ = gpu::gpu_fq12_sqr_chain(&input_limbs, iters);

    let start = Instant::now();
    for _ in 0..trials {
        let _ = std::hint::black_box(gpu::gpu_fq12_sqr_chain(&input_limbs, iters));
    }
    let gpu_time = start.elapsed() / trials;

    let start = Instant::now();
    for _ in 0..trials {
        let _: Vec<Fq12> = std::hint::black_box(
            inputs
                .iter()
                .map(|f| {
                    let mut val = *f;
                    for _ in 0..iters {
                        val = val.square();
                    }
                    val
                })
                .collect(),
        );
    }
    let cpu_single = start.elapsed() / trials;

    use rayon::prelude::*;
    let start = Instant::now();
    for _ in 0..trials {
        let _: Vec<Fq12> = std::hint::black_box(
            inputs
                .par_iter()
                .map(|f| {
                    let mut val = *f;
                    for _ in 0..iters {
                        val = val.square();
                    }
                    val
                })
                .collect(),
        );
    }
    let cpu_parallel = start.elapsed() / trials;

    println!("\n=== Fq12 Miller Loop Scale ({n} threads × {iters} Fq12 squarings) ===");
    println!("CPU single-threaded: {cpu_single:?}");
    println!("CPU parallel (rayon): {cpu_parallel:?}");
    println!("GPU (Metal):          {gpu_time:?}");
    let ratio_st = cpu_single.as_nanos() as f64 / gpu_time.as_nanos() as f64;
    let ratio_mt = cpu_parallel.as_nanos() as f64 / gpu_time.as_nanos() as f64;
    println!("vs single-thread:     {ratio_st:.1}x");
    println!("vs parallel (rayon):  {ratio_mt:.1}x");
}

#[test]
fn test_verify_metal_constants() {
    use ark_ec::short_weierstrass::SWCurveConfig;

    let two_inv = Fq::from(2u64).inverse().unwrap();
    assert_eq!(
        fq_to_limbs(&two_inv),
        [
            0x4f060572, 0x87bee7d2, 0x2f1c6ae5, 0xd0fd2add, 0xfcfd4f44, 0x8f5f7492, 0x3d9cbfac,
            0x1f37631a
        ],
        "TWO_INV"
    );

    let b = <ark_bn254::g2::Config as SWCurveConfig>::COEFF_B;
    assert_eq!(
        fq_to_limbs(&b.c0),
        [
            0x77b802a8, 0x3bf938e3, 0x3633535d, 0x020b1b27, 0x49755260, 0x26b7edf0, 0x4384a86d,
            0x2514c632
        ],
        "COEFF_B.c0"
    );
    assert_eq!(
        fq_to_limbs(&b.c1),
        [
            0xd1dcff67, 0x38e7eccc, 0x93ce0d3e, 0x65f0b37d, 0x22ac00aa, 0xd749d0dd, 0x4a688d4d,
            0x0141b9ce
        ],
        "COEFF_B.c1"
    );

    let tx = <ark_bn254::Config as BnConfig>::TWIST_MUL_BY_Q_X;
    let ty = <ark_bn254::Config as BnConfig>::TWIST_MUL_BY_Q_Y;
    assert_eq!(
        fq_to_limbs(&tx.c0),
        [
            0x4563ab30, 0xb5773b10, 0xa9aa6454, 0x347f91c8, 0x242e0991, 0x7a007127, 0x118214ec,
            0x1956bcd8
        ],
        "TWIST_X.c0"
    );
    assert_eq!(
        fq_to_limbs(&tx.c1),
        [
            0xa0aa4757, 0x6e849f1e, 0x89f89141, 0xaa1c7b6d, 0xfae0ca3a, 0xb6e713cd, 0x4e82ebc3,
            0x26694fbb
        ],
        "TWIST_X.c1"
    );
    assert_eq!(
        fq_to_limbs(&ty.c0),
        [
            0x2936b629, 0xe4bbdd0c, 0xe133bacb, 0xbb30f162, 0xf9645366, 0x31a9d1b6, 0xa500f8dd,
            0x253570be
        ],
        "TWIST_Y.c0"
    );
    assert_eq!(
        fq_to_limbs(&ty.c1),
        [
            0x5ffe77c7, 0xa1d77ce4, 0x7826d1db, 0x07affd11, 0xbb7edc6b, 0x6d16bd27, 0x85defecc,
            0x2c872002
        ],
        "TWIST_Y.c1"
    );

    let ark_lc = <ark_bn254::Config as BnConfig>::ATE_LOOP_COUNT;
    let metal_lc: [i8; 65] = [
        0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0,
        0, -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, -1, 0,
        -1, 0, 0, 0, 1, 0, 1, 1,
    ];
    for i in 0..ark_lc.len() {
        assert_eq!(ark_lc[i], metal_lc[i], "ATE_LOOP_COUNT[{i}]");
    }
    println!("All Metal constants verified against arkworks");
}

#[test]
fn test_gpu_miller_loop_correctness() {
    use ark_bn254::{Bn254, G1Affine, G1Projective, G2Affine, G2Projective};
    use ark_ec::{pairing::Pairing, CurveGroup};

    let mut rng = ark_std::test_rng();
    let n = 4;

    let g1_points: Vec<G1Affine> = (0..n)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    let g2_points: Vec<G2Affine> = (0..n)
        .map(|_| G2Projective::rand(&mut rng).into_affine())
        .collect();

    // Reference: arkworks individual Miller loops
    let expected: Vec<Fq12> = g1_points
        .iter()
        .zip(&g2_points)
        .map(|(p, q)| Bn254::multi_miller_loop([*p], [*q]).0)
        .collect();

    // Convert to GPU format: G1Affine = 16 u32 (x, y), G2Affine = 32 u32 (x.c0, x.c1, y.c0, y.c1)
    let gpu_g1: Vec<[u32; 16]> = g1_points
        .iter()
        .map(|p| {
            let mut limbs = [0u32; 16];
            limbs[..8].copy_from_slice(&fq_to_limbs(&p.x));
            limbs[8..].copy_from_slice(&fq_to_limbs(&p.y));
            limbs
        })
        .collect();
    let gpu_g2: Vec<[u32; 32]> = g2_points
        .iter()
        .map(|p| {
            let mut limbs = [0u32; 32];
            limbs[..8].copy_from_slice(&fq_to_limbs(&p.x.c0));
            limbs[8..16].copy_from_slice(&fq_to_limbs(&p.x.c1));
            limbs[16..24].copy_from_slice(&fq_to_limbs(&p.y.c0));
            limbs[24..].copy_from_slice(&fq_to_limbs(&p.y.c1));
            limbs
        })
        .collect();

    let gpu_results = gpu::gpu_miller_loop(&gpu_g1, &gpu_g2);

    for i in 0..n {
        let gpu_fq12 = limbs_to_fq12(&gpu_results[i]);
        assert_eq!(expected[i], gpu_fq12, "Miller loop mismatch at pair {i}");
    }
    println!("GPU Miller loop: all {n} pairs match arkworks");
}

#[test]
fn test_gpu_dory_integration() {
    use ark_bn254::{Bn254, G1Affine, G1Projective, G2Affine, G2Projective};
    use ark_ec::{pairing::Pairing, CurveGroup};

    // Register the Metal GPU Miller loop with dory-pcs
    super::register();

    let mut rng = ark_std::test_rng();
    let n = 64;

    let g1_points: Vec<G1Affine> = (0..n)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    let g2_points: Vec<G2Affine> = (0..n)
        .map(|_| G2Projective::rand(&mut rng).into_affine())
        .collect();

    // Reference: arkworks multi_miller_loop + final_exponentiation (the full pairing)
    let prepared_g1: Vec<<Bn254 as Pairing>::G1Prepared> =
        g1_points.iter().map(|p| (*p).into()).collect();
    let prepared_g2: Vec<<Bn254 as Pairing>::G2Prepared> =
        g2_points.iter().map(|p| (*p).into()).collect();
    let expected = Bn254::multi_miller_loop(prepared_g1, prepared_g2);
    let expected = Bn254::final_exponentiation(expected).unwrap();

    // Now compute through the dory-pcs pairing API (which should use GPU)
    use crate::poly::commitment::dory::wrappers::{ArkG1, ArkG2};
    let dory_g1: Vec<ArkG1> = g1_points.iter().map(|p| ArkG1((*p).into())).collect();
    let dory_g2: Vec<ArkG2> = g2_points.iter().map(|p| ArkG2((*p).into())).collect();

    use dory::primitives::arithmetic::PairingCurve;
    let gpu_result = dory::backends::arkworks::BN254::multi_pair(&dory_g1, &dory_g2);

    assert_eq!(
        expected.0, gpu_result.0,
        "GPU dory-pcs integration mismatch"
    );
    println!("GPU dory-pcs integration: {n}-pair multi_pairing matches arkworks");
}

#[test]
fn bench_gpu_miller_loop_crossover() {
    use ark_bn254::{Bn254, G1Affine, G1Projective, G2Affine, G2Projective};
    use ark_ec::{pairing::Pairing, CurveGroup};

    let mut rng = ark_std::test_rng();

    for n in [32, 64, 128, 256, 512, 1024, 2048] {
        let g1: Vec<G1Affine> = (0..n)
            .map(|_| G1Projective::rand(&mut rng).into_affine())
            .collect();
        let g2: Vec<G2Affine> = (0..n)
            .map(|_| G2Projective::rand(&mut rng).into_affine())
            .collect();

        let trials = if n <= 128 { 10 } else { 3 };

        // GPU path: conversion + kernel + Fq12 reduction + final exp
        let g1_limbs: Vec<_> = g1
            .iter()
            .map(|p| {
                let mut l = [0u32; 16];
                l[..8].copy_from_slice(&fq_to_limbs(&p.x));
                l[8..].copy_from_slice(&fq_to_limbs(&p.y));
                l
            })
            .collect();
        let g2_limbs: Vec<_> = g2
            .iter()
            .map(|p| {
                let mut l = [0u32; 32];
                l[..8].copy_from_slice(&fq_to_limbs(&p.x.c0));
                l[8..16].copy_from_slice(&fq_to_limbs(&p.x.c1));
                l[16..24].copy_from_slice(&fq_to_limbs(&p.y.c0));
                l[24..].copy_from_slice(&fq_to_limbs(&p.y.c1));
                l
            })
            .collect();

        // Warmup
        let _ = gpu::gpu_miller_loop(&g1_limbs, &g2_limbs);

        let start = Instant::now();
        for _ in 0..trials {
            let results = gpu::gpu_miller_loop(&g1_limbs, &g2_limbs);
            let combined_fq12: Fq12 = results
                .iter()
                .map(limbs_to_fq12)
                .fold(ark_ff::One::one(), |a, r| a * r);
            let combined = ark_ec::pairing::MillerLoopOutput(combined_fq12);
            let _ = std::hint::black_box(Bn254::final_exponentiation(combined).unwrap());
        }
        let gpu_time = start.elapsed() / trials as u32;

        // CPU parallel path (mimics dory-pcs multi_pair_parallel)
        use rayon::prelude::*;
        let start = Instant::now();
        for _ in 0..trials {
            let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> =
                g1.iter().map(|&a| a.into()).collect();
            let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> =
                g2.iter().map(|&a| a.into()).collect();

            let chunk_size = std::cmp::max(128, n / rayon::current_num_threads());
            let combined = ps_prep
                .par_chunks(chunk_size)
                .zip(qs_prep.par_chunks(chunk_size))
                .map(|(ps_c, qs_c)| Bn254::multi_miller_loop(ps_c.to_vec(), qs_c.to_vec()))
                .reduce(
                    || ark_ec::pairing::MillerLoopOutput(ark_ff::One::one()),
                    |a, b| ark_ec::pairing::MillerLoopOutput(a.0 * b.0),
                );
            let _ = std::hint::black_box(Bn254::final_exponentiation(combined).unwrap());
        }
        let cpu_time = start.elapsed() / trials as u32;

        let ratio = cpu_time.as_micros() as f64 / gpu_time.as_micros() as f64;
        let winner = if ratio > 1.0 { "GPU" } else { "CPU" };
        println!(
            "n={n:5}: GPU={gpu_time:>8?}  CPU={cpu_time:>8?}  ratio={ratio:.2}x  → {winner} wins"
        );
    }
}
