//! Debugging tests for zkMetal CPU Pippenger MSM interop.
//!
//! This module helps verify the FFI boundary between arkworks and zkMetal's C code.

use ark_bn254::{Fq, G1Affine, G1Projective, Fr};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_ff::{BigInt, One, PrimeField, UniformRand};
use ark_std::test_rng;
use rand::Rng;

/// Test with multiple random points and scalars
#[cfg(feature = "zkmetal")]
#[test]
fn test_zkmetal_multi_point_msm() {
    // Test 1: Single point with scalar=1 should equal G
    let generator_affine = G1Projective::generator().into_affine();
    let points1: Vec<G1Affine> = vec![generator_affine];
    let scalars1: Vec<Fr> = vec![Fr::from(1)];

    // Test 2: Two points with scalar=1 each should equal 2*G
    let points2: Vec<G1Affine> = vec![generator_affine, generator_affine];
    let scalars2: Vec<Fr> = vec![Fr::from(1), Fr::from(1)];

    // Test 3: Non-trivial scalars to verify Montgomery->Standard conversion
    // Fr(2) and Fr(3) have different Montgomery vs standard representations
    let scalars3: Vec<Fr> = vec![Fr::from(2), Fr::from(3)];

    // Compute reference with arkworks
    use ark_ec::scalar_mul::variable_base::VariableBaseMSM;

    let ark1: G1Projective = VariableBaseMSM::msm(&points1, &scalars1).expect("msm should not fail");
    let ark1_affine = ark1.into_affine();
    let ark1_x: BigInt<4> = ark1_affine.x.into_bigint();
    let ark1_y: BigInt<4> = ark1_affine.y.into_bigint();
    println!("\n=== Arkworks 1 point * 1 (should be G): {:?} ===", (ark1_x.0[0], ark1_y.0[0]));

    let ark2: G1Projective = VariableBaseMSM::msm(&points2, &scalars2).expect("msm should not fail");
    let ark2_affine = ark2.into_affine();
    let ark2_x: BigInt<4> = ark2_affine.x.into_bigint();
    let ark2_y: BigInt<4> = ark2_affine.y.into_bigint();
    println!("=== Arkworks 2 points * 1 (should be 2G): {:?} ===", (ark2_x.0[0], ark2_y.0[0]));

    let ark3: G1Projective = VariableBaseMSM::msm(&points2, &scalars3).expect("msm should not fail");
    let ark3_affine = ark3.into_affine();
    let ark3_x: BigInt<4> = ark3_affine.x.into_bigint();
    let ark3_y: BigInt<4> = ark3_affine.y.into_bigint();
    println!("=== Arkworks 2*G + 3*G = 5G: {:?} ===", (ark3_x.0[0], ark3_y.0[0]));
    println!("    = 2*G + 3*G affine: x={:?}, y={:?}", ark3_affine.x, ark3_affine.y);

    // Now test with zkMetal
    let zk1 = super::pippenger_msm(&points1, &scalars1);
    let zk1_affine = zk1.into_affine();
    let zk1_x: BigInt<4> = zk1_affine.x.into_bigint();
    let zk1_y: BigInt<4> = zk1_affine.y.into_bigint();
    println!("\n=== zkMetal 1 point * 1: {:?} ===", (zk1_x.0[0], zk1_y.0[0]));

    let zk2 = super::pippenger_msm(&points2, &scalars2);
    let zk2_affine = zk2.into_affine();
    let zk2_x: BigInt<4> = zk2_affine.x.into_bigint();
    let zk2_y: BigInt<4> = zk2_affine.y.into_bigint();
    println!("=== zkMetal 2 points * 1: {:?} ===", (zk2_x.0[0], zk2_y.0[0]));

    let zk3 = super::pippenger_msm(&points2, &scalars3);
    let zk3_affine = zk3.into_affine();
    let zk3_x: BigInt<4> = zk3_affine.x.into_bigint();
    let zk3_y: BigInt<4> = zk3_affine.y.into_bigint();
    println!("=== zkMetal 2G + 3G: {:?} ===", (zk3_x.0[0], zk3_y.0[0]));

    // Print expected vs actual for debugging
    println!("\nExpected 2G: {:?}", (ark2_x.0[0], ark2_y.0[0]));
    println!("Got zkMetal: {:?}", (zk2_x.0[0], zk2_y.0[0]));
    println!("\nExpected 2G+3G: {:?}", (ark3_x.0[0], ark3_y.0[0]));
    println!("Got zkMetal: {:?}", (zk3_x.0[0], zk3_y.0[0]));

    // Verify both match arkworks
    let match1 = ark1_x.0 == zk1_x.0 && ark1_y.0 == zk1_y.0;
    let match2 = ark2_x.0 == zk2_x.0 && ark2_y.0 == zk2_y.0;
    let match3 = ark3_x.0 == zk3_x.0 && ark3_y.0 == zk3_y.0;
    println!("\n1 point match: {}", match1);
    println!("2 point (1,1) match: {}", match2);
    println!("2 point (2,3) match: {}", match3);

    if !match2 {
        println!("zkMetal 2*1 mismatch - this indicates a bug in zkMetal or input conversion");
    }
    if !match3 {
        println!("zkMetal 2G+3G mismatch - this indicates a bug in Montgomery->Standard scalar conversion");
    }
}

/// Direct zkMetal FFI test to isolate the issue
#[cfg(feature = "zkmetal")]
#[test]
fn test_zkmetal_direct_ffi() {
    fn ark_to_zk_fmt(point: &G1Affine) -> [u64; 8] {
        let x_bigint: [u64; 4] = unsafe { std::mem::transmute_copy(&point.x.into_bigint()) };
        let y_bigint: [u64; 4] = unsafe { std::mem::transmute_copy(&point.y.into_bigint()) };
        [x_bigint[0], x_bigint[1], x_bigint[2], x_bigint[3],
         y_bigint[0], y_bigint[1], y_bigint[2], y_bigint[3]]
    }

    fn fr_to_zk_scalar(fr: &Fr) -> [u32; 8] {
        let bigint: BigInt<4> = fr.into_bigint();
        let mut result = [0u32; 8];
        for (i, limb) in bigint.0.iter().enumerate() {
            result[i * 2] = *limb as u32;
            result[i * 2 + 1] = (*limb >> 32) as u32;
        }
        result
    }

    let generator_affine = G1Projective::generator().into_affine();
    let points = vec![generator_affine, generator_affine];
    let scalars = vec![Fr::from(1), Fr::from(1)];

    // Convert to zkMetal format
    let mut points_raw: Vec<u64> = Vec::with_capacity(16);
    for p in &points {
        let zk_fmt = ark_to_zk_fmt(p);
        points_raw.extend_from_slice(&zk_fmt);
    }

    let mut scalars_raw: Vec<u32> = Vec::with_capacity(16);
    for s in &scalars {
        let zk_scalar = fr_to_zk_scalar(s);
        scalars_raw.extend_from_slice(&zk_scalar);
    }

    println!("Input points: {:?}", points_raw);
    println!("Input scalars: {:?}", scalars_raw);

    // Call zkMetal directly
    let mut result: [u64; 12] = [0; 12];
    unsafe {
        bn254_pippenger_msm(
            points_raw.as_ptr(),
            scalars_raw.as_ptr(),
            2,
            result.as_mut_ptr(),
        );
    }

    println!("zkMetal projective result: {:?}", result);

    // Reference with arkworks
    use ark_ec::scalar_mul::variable_base::VariableBaseMSM;
    let ark_result: G1Projective = VariableBaseMSM::msm(&points, &scalars).expect("msm should not fail");
    let ark_affine = ark_result.into_affine();
    let ark_x: BigInt<4> = ark_affine.x.into_bigint();
    let ark_y: BigInt<4> = ark_affine.y.into_bigint();
    println!("arkworks affine: x={:?}, y={:?}", ark_x.0, ark_y.0);
}

use zkmetal_sys::{bn254_pippenger_msm, bn254_msm_projective, bn254_projective_to_affine};

/// Convert arkworks G1Affine to zkMetal format (8 u64 per point)
fn ark_to_zk_format(point: &G1Affine) -> [u64; 8] {
    // arkworks stores in Montgomery form as [u64; 4]
    let x_bigint: [u64; 4] = unsafe { std::mem::transmute_copy(&point.x.into_bigint()) };
    let y_bigint: [u64; 4] = unsafe { std::mem::transmute_copy(&point.y.into_bigint()) };
    [x_bigint[0], x_bigint[1], x_bigint[2], x_bigint[3],
     y_bigint[0], y_bigint[1], y_bigint[2], y_bigint[3]]
}

/// Convert arkworks Fr to zkMetal scalar format (8 u32 per scalar)
fn fr_to_zk_scalar(fr: &Fr) -> [u32; 8] {
    let bigint: BigInt<4> = fr.into_bigint();
    // Convert [u64; 4] to [u32; 8] - little-endian
    let mut result = [0u32; 8];
    for (i, limb) in bigint.0.iter().enumerate() {
        result[i * 2] = *limb as u32;
        result[i * 2 + 1] = (*limb >> 32) as u32;
    }
    result
}

/// Call zkMetal directly via FFI
#[cfg(feature = "zkmetal")]
fn call_zkmetal(points: &[G1Affine], scalars: &[Fr]) -> (G1Projective, [u64; 12]) {
    let n = points.len();
    assert_eq!(n, scalars.len());

    // Convert points to zkMetal format
    let mut points_raw: Vec<u64> = Vec::with_capacity(n * 8);
    for p in points {
        let zk_fmt = ark_to_zk_format(p);
        points_raw.extend_from_slice(&zk_fmt);
    }

    // Convert scalars to zkMetal format
    let mut scalars_raw: Vec<u32> = Vec::with_capacity(n * 8);
    for s in scalars {
        let zk_scalar = fr_to_zk_scalar(s);
        scalars_raw.extend_from_slice(&zk_scalar);
    }

    // Call zkMetal
    let mut result: [u64; 12] = [0; 12];
    println!("Calling zkMetal FFI...");
    unsafe {
        bn254_pippenger_msm(
            points_raw.as_ptr(),
            scalars_raw.as_ptr(),
            n as i32,
            result.as_mut_ptr(),
        );
    }
    println!("zkMetal returned, raw u64[12]: {:?}", result);

    // Convert result back - these are in standard projective form
    // zkMetal outputs (X, Y, Z) in Montgomery form, but for BN254 G1,
    // these are field elements in the base field Fq (same as Fr for scalar field)
    let x_arr: [u64; 4] = [result[0], result[1], result[2], result[3]];
    let y_arr: [u64; 4] = [result[4], result[5], result[6], result[7]];
    let z_arr: [u64; 4] = [result[8], result[9], result[10], result[11]];
    println!("After slice: X={:?}, Y={:?}, Z={:?}", x_arr, y_arr, z_arr);

    // For BN254 G1, we need Fq types, not Fr
    // The Fq and Fr are both 254-bit fields on BN254, so they have the same representation
    use ark_bn254::Fq;
    let x: Fq = unsafe { std::mem::transmute_copy(&x_arr) };
    let y: Fq = unsafe { std::mem::transmute_copy(&y_arr) };
    let z: Fq = unsafe { std::mem::transmute_copy(&z_arr) };
    println!("After transmute to Fq: x={}, y={}, z={}", x, y, z);

    let z_proj = G1Projective::new(x, y, z);

    (z_proj, result)
}

/// Compute reference using arkworks
fn arkworks_msm(points: &[G1Affine], scalars: &[Fr]) -> G1Projective {
    use ark_ec::scalar_mul::variable_base::VariableBaseMSM;
    VariableBaseMSM::msm(points, scalars).expect("msm should not fail")
}

#[test]
fn test_scalar_conversion() {
    // Test scalar format
    let mut rng = test_rng();
    let scalar = Fr::rand(&mut rng);

    let bigint: BigInt<4> = scalar.into_bigint();
    println!("\n=== Scalar Conversion Test ===");
    println!("scalar BigInt limbs: {:?}", bigint.0);

    let zk_scalar = fr_to_zk_scalar(&scalar);
    println!("zk_scalar bytes (LE u32): {:?}", zk_scalar);

    // The BN254 scalar field is ~254 bits, so we should have significant data in all 8 limbs
    let non_zero_limbs = zk_scalar.iter().filter(|&&x| x != 0).count();
    println!("Non-zero limbs: {}/8", non_zero_limbs);

    // Also check what the raw BigInt bytes look like
    println!("BigInt raw bytes: {:?}", unsafe {
        std::mem::transmute::<BigInt<4>, [u8; 32]>(bigint)
    });
}

#[test]
fn test_point_conversion() {
    // Test point format
    let mut rng = test_rng();
    let point = G1Projective::rand(&mut rng).into_affine();

    println!("\n=== Point Conversion Test ===");

    let zk_fmt = ark_to_zk_format(&point);
    println!("zk_point x limbs: {:?}", &zk_fmt[0..4]);
    println!("zk_point y limbs: {:?}", &zk_fmt[4..8]);

    // BN254 x coordinate should be ~254 bits, spread across 4 limbs
    let x_non_zero = zk_fmt[0..4].iter().filter(|&&x| x != 0).count();
    let y_non_zero = zk_fmt[4..8].iter().filter(|&&x| x != 0).count();
    println!("X non-zero limbs: {}/4", x_non_zero);
    println!("Y non-zero limbs: {}/4", y_non_zero);

    // Print the raw bytes
    println!("X bytes: {:?}", unsafe {
        std::mem::transmute::<[u64; 4], [u8; 32]>([zk_fmt[0], zk_fmt[1], zk_fmt[2], zk_fmt[3]])
    });
}

#[test]
fn test_known_scalar_msm() {
    // Test with a known case: 1*G = G for generator
    let generator_affine = G1Projective::generator().into_affine();
    let one = Fr::from(1);

    println!("\n=== Known Scalar MSM Test ===");
    println!("Generator: x={}, y={}", generator_affine.x, generator_affine.y);

    let ark_result = arkworks_msm(&[generator_affine], &[one]);
    let ark_affine = ark_result.into_affine();

    println!("1*G affine: x={}, y={}", ark_affine.x, ark_affine.y);

    // The result should be the generator itself
    assert_eq!(ark_affine.x, generator_affine.x, "X coordinate mismatch");
    assert_eq!(ark_affine.y, generator_affine.y, "Y coordinate mismatch");
    println!("SUCCESS: 1*G == G");
}

#[test]
fn test_scalar_one_conversion() {
    // Test what Fr::from(1) looks like in zkMetal format
    let one = Fr::from(1);
    let bigint: BigInt<4> = one.into_bigint();

    println!("\n=== Fr(1) Conversion ===");
    println!("Fr(1) BigInt: {:?}", bigint.0);

    let zk_scalar = fr_to_zk_scalar(&one);
    println!("Fr(1) zk format: {:?}", zk_scalar);

    // For Fr(1), we expect only the first limb to be set (lowest 32 bits = 1)
    assert_eq!(zk_scalar[0], 1, "First limb should be 1");
    for i in 1..8 {
        assert_eq!(zk_scalar[i], 0, "All other limbs should be 0");
    }
    println!("SUCCESS: Fr(1) correctly formatted");
}

#[cfg(feature = "zkmetal")]
#[test]
fn test_zkmetal_input_output() {
    // Debug: verify input and output of zkMetal

    let generator_affine = G1Projective::generator().into_affine();
    let one = Fr::from(1);

    println!("\n=== zkMetal Input/Output Debug ===");

    // Print generator x in different forms
    let gen_x_bigint: BigInt<4> = generator_affine.x.into_bigint();
    println!("Generator x standard form BigInt: {:?}", gen_x_bigint.0);
    println!("Generator x Fq display: {}", generator_affine.x);

    // The Fq is in Montgomery form. In Montgomery form:
    // standard_value * R mod p = Montgomery_value
    // So Montgomery(1) = 1 * R mod p

    // Let's compute what R mod p should be
    // For BN254, p = 0x30644e72e131a029b85045b68181585d
    // R = 2^256
    // We need R mod p

    // Let's just check: if gen_x = 1 (standard), then Montgomery(gen_x) = gen_x * R mod p
    // This means BigInt(gen_x) should NOT be [1, 0, 0, 0] if it's in Montgomery form!

    // Wait, maybe arkworks is storing standard form and printing standard form?
    // Let me check by comparing with what we pass to zkMetal

    // What we pass to zkMetal:
    fn ark_to_zk_fmt(point: &G1Affine) -> [u64; 8] {
        let x_bigint: [u64; 4] = unsafe { std::mem::transmute_copy(&point.x.into_bigint()) };
        let y_bigint: [u64; 4] = unsafe { std::mem::transmute_copy(&point.y.into_bigint()) };
        [x_bigint[0], x_bigint[1], x_bigint[2], x_bigint[3],
         y_bigint[0], y_bigint[1], y_bigint[2], y_bigint[3]]
    }
    let zk_fmt = ark_to_zk_fmt(&generator_affine);
    println!("\nWhat we send to zkMetal:");
    println!("  X limbs: {:?}", &zk_fmt[0..4]);
    println!("  Y limbs: {:?}", &zk_fmt[4..8]);

    // This is what ark_to_zk_format produces - these should be the Montgomery form of x and y
    // If gen_x is Montgomery(1), then zk_fmt[0] should equal R mod p in little-endian
    // R mod p should be approximately 0x16f20404b2f24275fcacd20f5bfbbfc1
    // But zk_fmt[0] is 1, not that!

    // So either: ark_to_zk_format is wrong, or arkworks is storing standard form
    // Let's verify with a known value
    let test_x = ark_bn254::Fq::from(123);
    let test_bigint: BigInt<4> = test_x.into_bigint();

    // Define local conversion
    fn local_ark_to_zk(point: &G1Affine) -> [u64; 8] {
        let x_bigint: [u64; 4] = unsafe { std::mem::transmute_copy(&point.x.into_bigint()) };
        let y_bigint: [u64; 4] = unsafe { std::mem::transmute_copy(&point.y.into_bigint()) };
        [x_bigint[0], x_bigint[1], x_bigint[2], x_bigint[3],
         y_bigint[0], y_bigint[1], y_bigint[2], y_bigint[3]]
    }
    let test_zk = local_ark_to_zk(&G1Affine::new_unchecked(test_x, test_x));

    println!("\nTest: Fq(123)");
    println!("  BigInt: {:?}", test_bigint.0);
    println!("  zk format X: {:?}", &test_zk[0..4]);

    // If BigInt and zk format are the same, then arkworks is in standard form
    // If they differ, arkworks might be in Montgomery form
}

// Old test_zkmetal_single_point moved to separate file
// Keeping the debug test for now

#[cfg(feature = "zkmetal")]
#[test]
fn test_zkmetal_raw_projective() {
    // Test zkMetal WITHOUT constructing arkworks G1Projective
    // Just look at the raw result from zkMetal

    let generator_affine = G1Projective::generator().into_affine();
    let one = Fr::from(1);

    println!("\n=== Raw Projective Test ===");

    // Use zkMetal-sys's neon FFI to call bn254_projective_to_affine
    extern "C" {
        fn bn254_projective_to_affine(p: *const u64, affine: *mut u64);
        fn bn254_batch_to_affine(proj: *const u64, aff: *mut u64, n: i32);
    }

    let n = 1;

    // Convert points to zkMetal format
    let mut points_raw: Vec<u64> = Vec::with_capacity(n * 8);
    let zk_fmt = ark_to_zk_format(&generator_affine);
    points_raw.extend_from_slice(&zk_fmt);

    // Convert scalars to zkMetal format
    let mut scalars_raw: Vec<u32> = Vec::with_capacity(n * 8);
    let zk_scalar = fr_to_zk_scalar(&one);
    scalars_raw.extend_from_slice(&zk_scalar);

    // Call zkMetal Pippenger
    let mut projective_result: [u64; 12] = [0; 12];
    unsafe {
        bn254_pippenger_msm(
            points_raw.as_ptr(),
            scalars_raw.as_ptr(),
            n as i32,
            projective_result.as_mut_ptr(),
        );
    }

    println!("zkMetal projective result:");
    println!("  X: {:?}", &projective_result[0..4]);
    println!("  Y: {:?}", &projective_result[4..8]);
    println!("  Z: {:?}", &projective_result[8..12]);

    // Now call zkMetal's projective to affine conversion
    let mut affine_result: [u64; 8] = [0; 8];
    unsafe {
        bn254_projective_to_affine(
            projective_result.as_ptr(),
            affine_result.as_mut_ptr(),
        );
    }

    println!("zkMetal affine result (from zkMetal's converter):");
    println!("  X: {:?}", &affine_result[0..4]);
    println!("  Y: {:?}", &affine_result[4..8]);

    // Compare with arkworks
    let ark_result = arkworks_msm(&[generator_affine], &[one]);
    let ark_affine = ark_result.into_affine();
    let ark_x_bigint: BigInt<4> = ark_affine.x.into_bigint();
    let ark_y_bigint: BigInt<4> = ark_affine.y.into_bigint();
    println!("arkworks affine:");
    println!("  X: {:?}", ark_x_bigint);
    println!("  Y: {:?}", ark_y_bigint);

    // Check if they match
    let affine_x_match = affine_result[0..4] == ark_x_bigint.0[..];
    let affine_y_match = affine_result[4..8] == ark_y_bigint.0[..];
    println!("Affine X matches: {}", affine_x_match);
    println!("Affine Y matches: {}", affine_y_match);

    if affine_x_match && affine_y_match {
        println!("SUCCESS: zkMetal affine matches arkworks!");
    } else {
        println!("MISMATCH: zkMetal affine does not match arkworks!");
    }
}

/// Test to verify scalar extraction and window extraction behavior
#[cfg(feature = "zkmetal")]
#[test]
fn test_zkmetal_scalar_extraction() {
    fn fr_to_zk_scalar(fr: &Fr) -> [u32; 8] {
        let bigint: BigInt<4> = fr.into_bigint();
        let mut result = [0u32; 8];
        for (i, limb) in bigint.0.iter().enumerate() {
            result[i * 2] = *limb as u32;
            result[i * 2 + 1] = (*limb >> 32) as u32;
        }
        result
    }

    // Test scalar extraction
    let one = Fr::from(1);
    let big_scalar = Fr::from(100);
    let larger = Fr::from(1000000);

    println!("\n=== Scalar Extraction Test ===");
    println!("Fr(1): {:?}", fr_to_zk_scalar(&one));
    println!("Fr(100): {:?}", fr_to_zk_scalar(&big_scalar));
    println!("Fr(1000000): {:?}", fr_to_zk_scalar(&larger));

    // Test with large random scalars
    let mut rng = test_rng();
    for i in 0..5 {
        let scalar = Fr::rand(&mut rng);
        let limbs = fr_to_zk_scalar(&scalar);
        println!("Random scalar {}: {:?}", i, limbs);
    }

    // Test window extraction manually
    // Simulate what extract_window does
    let scalar = fr_to_zk_scalar(&one);
    println!("\n=== Simulated Window Extraction ===");
    for wb in [3, 5, 8] {
        println!("\nWindow bits = {}", wb);
        let num_windows = (256 + wb - 1) / wb;
        for w in 0..num_windows.min(10) {
            let bit_offset = w * wb;
            let word_idx = bit_offset / 32;
            let bit_in_word = bit_offset % 32;

            let mut word = scalar[word_idx] as u64;
            if word_idx + 1 < 8 {
                word |= (scalar[word_idx + 1] as u64) << 32;
            }

            let digit = (word >> bit_in_word) & ((1u64 << wb) - 1);
            println!("  Window {}: bit_offset={}, word_idx={}, digit={}",
                w, bit_offset, word_idx, digit);
        }
    }
}
