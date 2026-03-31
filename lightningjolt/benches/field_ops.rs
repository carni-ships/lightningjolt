//! Microbenchmarks for BN254 field arithmetic on Apple Silicon.
//!
//! Measures throughput of the operations that dominate Jolt proving time:
//! - Field multiplication (Montgomery form)
//! - Field addition
//! - Polynomial evaluation
//! - MSM-relevant operations

use ark_bn254::Fr;
use ark_ff::{BigInteger, Field, PrimeField};
use ark_std::UniformRand;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

// Import our optimized Montgomery multiply
#[path = "../src/aarch64_mont.rs"]
mod aarch64_mont;

fn bench_field_mul(c: &mut Criterion) {
    let mut rng = ark_std::test_rng();
    let a = Fr::rand(&mut rng);
    let b = Fr::rand(&mut rng);

    let mut group = c.benchmark_group("field_mul");
    group.throughput(Throughput::Elements(1));
    group.bench_function("bn254_fr_mul", |bench| {
        bench.iter(|| black_box(a) * black_box(b))
    });
    group.finish();
}

fn bench_field_add(c: &mut Criterion) {
    let mut rng = ark_std::test_rng();
    let a = Fr::rand(&mut rng);
    let b = Fr::rand(&mut rng);

    let mut group = c.benchmark_group("field_add");
    group.throughput(Throughput::Elements(1));
    group.bench_function("bn254_fr_add", |bench| {
        bench.iter(|| black_box(a) + black_box(b))
    });
    group.finish();
}

fn bench_field_inv(c: &mut Criterion) {
    let mut rng = ark_std::test_rng();
    let a = Fr::rand(&mut rng);

    let mut group = c.benchmark_group("field_inv");
    group.throughput(Throughput::Elements(1));
    group.bench_function("bn254_fr_inv", |bench| {
        bench.iter(|| black_box(a).inverse())
    });
    group.finish();
}

fn bench_poly_eval(c: &mut Criterion) {
    let mut rng = ark_std::test_rng();
    let sizes = [64, 256, 1024, 4096];

    let mut group = c.benchmark_group("poly_eval");
    for &size in &sizes {
        let coeffs: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
        let point = Fr::rand(&mut rng);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_function(format!("horner_{size}"), |bench| {
            bench.iter(|| {
                // Horner's method
                let mut result = Fr::from(0u64);
                for c in coeffs.iter().rev() {
                    result = result * point + c;
                }
                black_box(result)
            })
        });
    }
    group.finish();
}

fn bench_parallel_sum(c: &mut Criterion) {
    use rayon::prelude::*;
    let mut rng = ark_std::test_rng();
    let size = 1 << 16; // 65536 elements
    let vals: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();

    let mut group = c.benchmark_group("parallel_sum");
    group.throughput(Throughput::Elements(size as u64));

    group.bench_function("sequential", |bench| {
        bench.iter(|| {
            let sum: Fr = vals.iter().copied().sum();
            black_box(sum)
        })
    });

    group.bench_function("rayon_par_iter", |bench| {
        bench.iter(|| {
            let sum: Fr = vals.par_iter().copied().sum();
            black_box(sum)
        })
    });

    group.finish();
}

fn bench_mont_mul_asm(c: &mut Criterion) {
    let mut rng = ark_std::test_rng();
    let a = Fr::rand(&mut rng);
    let b = Fr::rand(&mut rng);
    let a_limbs: [u64; 4] = a.into_bigint().0;
    let b_limbs: [u64; 4] = b.into_bigint().0;

    let mut group = c.benchmark_group("mont_mul_comparison");
    group.throughput(Throughput::Elements(1));

    group.bench_function("arkworks_generic", |bench| {
        bench.iter(|| {
            let mut x = black_box(a);
            x *= black_box(b);
            x
        })
    });

    group.bench_function("aarch64_asm", |bench| {
        bench.iter(|| {
            let mut result = [0u64; 4];
            aarch64_mont::mont_mul_4(black_box(&a_limbs), black_box(&b_limbs), &mut result);
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_field_mul,
    bench_field_add,
    bench_field_inv,
    bench_poly_eval,
    bench_parallel_sum,
    bench_mont_mul_asm,
);
criterion_main!(benches);
