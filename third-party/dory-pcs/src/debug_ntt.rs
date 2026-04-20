//! Debug script for NTT implementation
use crate::backends::ntt::cpu_ntt::TwiddleCache;
use crate::backends::ntt::fp128::Fp128Element;

fn main() {
    let log_n = 2;
    let cache = TwiddleCache::new(log_n);

    // Test with simple input [1, 0, 0, 0]
    let input: Vec<Fp128Element> = vec![
        Fp128Element::from_raw(1),
        Fp128Element::zero(),
        Fp128Element::zero(),
        Fp128Element::zero(),
    ];

    println!("Input: {:?}", input);

    // Forward NTT
    let fwd = cache.ntt_fwd(&input, log_n);
    println!("Forward NTT: {:?}", fwd);

    // Inverse NTT
    let inv = cache.ntt_inv(&fwd, log_n);
    println!("Inverse NTT: {:?}", inv);

    // Check roundtrip
    println!("\nRoundtrip check:");
    for (i, (a, b)) in input.iter().zip(inv.iter()).enumerate() {
        println!("  input[{}] = {:?}, recovered = {:?}, equal = {}", i, a, b, a == b);
    }

    // Check primitive root
    println!("\nPrimitive root verification (log_n={}):", log_n);
    let omega = Fp128Element::primitive_root_of_unity(log_n);
    let n = 1u64 << log_n;
    let omega_n = omega.pow(n);
    let omega_half = omega.pow(n / 2);
    println!("  omega = {:?}", omega);
    println!("  omega^n = {:?} (should be 1)", omega_n);
    println!("  omega^(n/2) = {:?} (should NOT be 1 for primitive)", omega_half);
    println!("  Is omega^n == 1? {}", omega_n.is_one());
    println!("  Is omega^(n/2) == 1? {} (should be false if primitive)", omega_half.is_one());

    // Check naive inverse
    println!("\nNaive inverse test:");
    let naive_inv = cache.ntt_inv_naive(&fwd, log_n);
    println!("Naive inverse: {:?}", naive_inv);
    for (i, (a, b)) in input.iter().zip(naive_inv.iter()).enumerate() {
        println!("  input[{}] = {:?}, recovered = {:?}, equal = {}", i, a, b, a == b);
    }
}