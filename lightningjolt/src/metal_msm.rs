//! Metal GPU acceleration for BN254 field operations.
//!
//! Uses Apple's Metal compute shaders for embarrassingly parallel
//! field arithmetic operations. Compiles shaders at runtime from
//! embedded .metal source (no Xcode required).
//!
//! Operations:
//! - `batch_field_mul`: element-wise field multiply of two vectors
//! - `batch_field_add`: element-wise field add of two vectors
//! - `scalar_vec_mul`: multiply all elements by a scalar
//! - `parallel_field_sum`: parallel reduction sum of a vector
//!
//! These operations underpin MSM (multi-scalar multiplication) and
//! polynomial evaluation — the main commitment bottlenecks in Jolt.

#[cfg(target_os = "macos")]
pub mod gpu {
    use metal::*;
    use std::mem;

    const SHADER_SOURCE: &str = include_str!("../shaders/bn254_field.metal");

    /// A 256-bit BN254 field element stored as 8 × u32 (matching the Metal shader layout).
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct GpuFp256 {
        pub limbs: [u32; 8],
    }

    impl GpuFp256 {
        /// Convert from 4 × u64 limbs (arkworks internal format) to 8 × u32
        pub fn from_u64_limbs(limbs: &[u64; 4]) -> Self {
            Self {
                limbs: [
                    limbs[0] as u32,
                    (limbs[0] >> 32) as u32,
                    limbs[1] as u32,
                    (limbs[1] >> 32) as u32,
                    limbs[2] as u32,
                    (limbs[2] >> 32) as u32,
                    limbs[3] as u32,
                    (limbs[3] >> 32) as u32,
                ],
            }
        }

        /// Convert back to 4 × u64 limbs
        pub fn to_u64_limbs(&self) -> [u64; 4] {
            [
                self.limbs[0] as u64 | (self.limbs[1] as u64) << 32,
                self.limbs[2] as u64 | (self.limbs[3] as u64) << 32,
                self.limbs[4] as u64 | (self.limbs[5] as u64) << 32,
                self.limbs[6] as u64 | (self.limbs[7] as u64) << 32,
            ]
        }

        pub fn zero() -> Self {
            Self { limbs: [0; 8] }
        }
    }

    /// Metal GPU compute context for BN254 field operations.
    pub struct MetalContext {
        device: Device,
        queue: CommandQueue,
        pipeline_mul: ComputePipelineState,
        pipeline_add: ComputePipelineState,
        pipeline_scalar_mul: ComputePipelineState,
        pipeline_sum: ComputePipelineState,
    }

    impl MetalContext {
        /// Initialize Metal context, compile shaders.
        pub fn new() -> Result<Self, String> {
            let device =
                Device::system_default().ok_or_else(|| "No Metal GPU device found".to_string())?;

            println!("  Metal GPU: {}", device.name());
            println!("  Unified memory: {}", device.has_unified_memory());
            println!(
                "  Max threadgroup size: {}",
                device.max_threads_per_threadgroup().width
            );

            let queue = device.new_command_queue();

            let opts = CompileOptions::new();
            let library = device
                .new_library_with_source(SHADER_SOURCE, &opts)
                .map_err(|e| format!("Shader compile error: {e}"))?;

            let make_pipeline =
                |dev: &Device, lib: &Library, name: &str| -> Result<ComputePipelineState, String> {
                    let func = lib
                        .get_function(name, None)
                        .map_err(|e| format!("Function '{name}' not found: {e}"))?;
                    dev.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| format!("Pipeline '{name}' error: {e}"))
                };

            let pipeline_mul = make_pipeline(&device, &library, "batch_field_mul")?;
            let pipeline_add = make_pipeline(&device, &library, "batch_field_add")?;
            let pipeline_scalar_mul = make_pipeline(&device, &library, "scalar_vec_mul")?;
            let pipeline_sum = make_pipeline(&device, &library, "parallel_field_sum")?;

            Ok(Self {
                device,
                queue,
                pipeline_mul,
                pipeline_add,
                pipeline_scalar_mul,
                pipeline_sum,
            })
        }

        /// Batch field multiply: out[i] = a[i] * b[i] mod p
        pub fn batch_mul(&self, a: &[GpuFp256], b: &[GpuFp256]) -> Vec<GpuFp256> {
            self.dispatch_binary(&self.pipeline_mul, a, b)
        }

        /// Generic binary dispatch for element-wise operations
        fn dispatch_binary(
            &self,
            pipeline: &ComputePipelineState,
            a: &[GpuFp256],
            b: &[GpuFp256],
        ) -> Vec<GpuFp256> {
            assert_eq!(a.len(), b.len());
            let n = a.len();
            let elem_size = mem::size_of::<GpuFp256>() as u64;

            let buf_a = self.device.new_buffer_with_data(
                a.as_ptr() as *const _,
                n as u64 * elem_size,
                MTLResourceOptions::StorageModeShared,
            );
            let buf_b = self.device.new_buffer_with_data(
                b.as_ptr() as *const _,
                n as u64 * elem_size,
                MTLResourceOptions::StorageModeShared,
            );
            let buf_out = self
                .device
                .new_buffer(n as u64 * elem_size, MTLResourceOptions::StorageModeShared);

            let cmd = self.queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&buf_a), 0);
            encoder.set_buffer(1, Some(&buf_b), 0);
            encoder.set_buffer(2, Some(&buf_out), 0);

            let grid = MTLSize::new(n as u64, 1, 1);
            let group = MTLSize::new(
                (pipeline.max_total_threads_per_threadgroup() as u64).min(n as u64),
                1,
                1,
            );
            encoder.dispatch_threads(grid, group);
            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            let ptr = buf_out.contents() as *const GpuFp256;
            let mut result = vec![GpuFp256::zero(); n];
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), n);
            }
            result
        }

        /// Batch field add: out[i] = a[i] + b[i] mod p
        pub fn batch_add(&self, a: &[GpuFp256], b: &[GpuFp256]) -> Vec<GpuFp256> {
            self.dispatch_binary(&self.pipeline_add, a, b)
        }

        /// Parallel reduction sum of a vector of field elements.
        pub fn parallel_sum(&self, input: &[GpuFp256]) -> GpuFp256 {
            let n = input.len();
            if n == 0 {
                return GpuFp256::zero();
            }

            let elem_size = mem::size_of::<GpuFp256>() as u64;
            let group_size = 256u64.min(n as u64);
            let num_groups = (n as u64 + group_size - 1) / group_size;

            let buf_in = self.device.new_buffer_with_data(
                input.as_ptr() as *const _,
                n as u64 * elem_size,
                MTLResourceOptions::StorageModeShared,
            );
            let buf_partial = self.device.new_buffer(
                num_groups * elem_size,
                MTLResourceOptions::StorageModeShared,
            );
            let count_val = n as u32;
            let buf_count = self.device.new_buffer_with_data(
                &count_val as *const u32 as *const _,
                4,
                MTLResourceOptions::StorageModeShared,
            );

            let cmd = self.queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline_sum);
            encoder.set_buffer(0, Some(&buf_in), 0);
            encoder.set_buffer(1, Some(&buf_partial), 0);
            encoder.set_buffer(2, Some(&buf_count), 0);

            let grid = MTLSize::new(num_groups * group_size, 1, 1);
            let group = MTLSize::new(group_size, 1, 1);
            encoder.dispatch_threads(grid, group);
            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // Read partial sums and reduce on CPU
            let ptr = buf_partial.contents() as *const GpuFp256;
            let partial: Vec<GpuFp256> = unsafe {
                let mut v = vec![GpuFp256::zero(); num_groups as usize];
                std::ptr::copy_nonoverlapping(ptr, v.as_mut_ptr(), num_groups as usize);
                v
            };

            // CPU reduction of partial sums (small number of groups)
            if partial.len() <= 256 {
                // Simple CPU reduce
                cpu_field_sum(&partial)
            } else {
                // Recursive GPU reduction
                self.parallel_sum(&partial)
            }
        }

        pub fn device_name(&self) -> String {
            self.device.name().to_string()
        }
    }

    /// CPU reference implementation of 32-bit word Montgomery multiply
    /// (mirrors the Metal shader algorithm exactly)
    fn cpu_mont_mul_32bit(a_64: &[u64; 4], b_64: &[u64; 4]) -> [u64; 4] {
        let modulus_32: [u32; 8] = [
            0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d, 0xb85045b6, 0xe131a029,
            0x30644e72,
        ];
        let inv32: u32 = 0xefffffff;

        // Convert inputs to 32-bit limbs
        let mut a = [0u32; 8];
        let mut b = [0u32; 8];
        for i in 0..4 {
            a[i * 2] = a_64[i] as u32;
            a[i * 2 + 1] = (a_64[i] >> 32) as u32;
            b[i * 2] = b_64[i] as u32;
            b[i * 2 + 1] = (b_64[i] >> 32) as u32;
        }

        // Schoolbook 256×256 → 512 bit multiply
        let mut product = [0u32; 16];
        for i in 0..8 {
            let mut carry = 0u32;
            for j in 0..8 {
                let uv = a[i] as u64 * b[j] as u64 + product[i + j] as u64 + carry as u64;
                product[i + j] = uv as u32;
                carry = (uv >> 32) as u32;
            }
            let prev = product[i + 8];
            product[i + 8] = prev.wrapping_add(carry);
            if product[i + 8] < carry {
                for k in (i + 9)..16 {
                    product[k] = product[k].wrapping_add(1);
                    if product[k] != 0 {
                        break;
                    }
                }
            }
        }

        // Montgomery reduction (word-by-word)
        let mut t = product;
        for i in 0..8 {
            let m = t[i].wrapping_mul(inv32);
            let mut carry = 0u32;
            for j in 0..8 {
                let uv = m as u64 * modulus_32[j] as u64 + t[i + j] as u64 + carry as u64;
                t[i + j] = uv as u32;
                carry = (uv >> 32) as u32;
            }
            for k in (i + 8)..16 {
                let s = t[k] as u64 + carry as u64;
                t[k] = s as u32;
                carry = (s >> 32) as u32;
                if carry == 0 {
                    break;
                }
            }
        }

        // Result is t[8..15], conditional subtract
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[i] = t[i + 8];
        }

        let mut sub = [0u32; 8];
        let mut borrow: i64 = 0;
        for i in 0..8 {
            let diff = result[i] as i64 - modulus_32[i] as i64 - borrow;
            sub[i] = diff as u32;
            borrow = if diff < 0 { 1 } else { 0 };
        }

        let out = if borrow != 0 { result } else { sub };

        // Convert back to 64-bit limbs
        [
            out[0] as u64 | (out[1] as u64) << 32,
            out[2] as u64 | (out[3] as u64) << 32,
            out[4] as u64 | (out[5] as u64) << 32,
            out[6] as u64 | (out[7] as u64) << 32,
        ]
    }

    /// CPU fallback for small reductions
    fn cpu_field_sum(elements: &[GpuFp256]) -> GpuFp256 {
        // Simple addition — for partial sum reduction only
        // Convert to u64, add, reduce mod p
        let modulus: [u64; 4] = [
            0x43e1f593f0000001,
            0x2833e84879b97091,
            0xb85045b68181585d,
            0x30644e72e131a029,
        ];

        let mut acc = [0u64; 4];
        for elem in elements {
            let limbs = elem.to_u64_limbs();
            let mut carry = 0u128;
            for i in 0..4 {
                carry += acc[i] as u128 + limbs[i] as u128;
                acc[i] = carry as u64;
                carry >>= 64;
            }
            // Reduce if >= p
            let mut borrow = 0i128;
            let mut sub = [0u64; 4];
            for i in 0..4 {
                borrow = acc[i] as i128 - modulus[i] as i128 - if borrow < 0 { 1 } else { 0 };
                sub[i] = borrow as u64;
            }
            if borrow >= 0 {
                acc = sub;
            }
        }
        GpuFp256::from_u64_limbs(&acc)
    }

    /// Run GPU benchmark: batch multiply N field elements
    pub fn benchmark_gpu(n: usize) -> Result<(), String> {
        use std::time::Instant;

        println!("\nMetal GPU Field Benchmark (n={n})");
        println!("===================================");

        let ctx = MetalContext::new()?;

        // Generate random-ish test data
        let a: Vec<GpuFp256> = (0..n)
            .map(|i| {
                GpuFp256::from_u64_limbs(&[
                    (i as u64).wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1),
                    (i as u64).wrapping_mul(0x517cc1b727220a95),
                    (i as u64).wrapping_mul(0x6c62272e07bb0142),
                    (i as u64 % 0x30644e72e131a029),
                ])
            })
            .collect();
        let b = a.clone();

        // Warmup
        let _ = ctx.batch_mul(&a[..64.min(n)], &b[..64.min(n)]);

        // Benchmark batch multiply
        let start = Instant::now();
        let _result = ctx.batch_mul(&a, &b);
        let mul_time = start.elapsed();
        let mul_throughput = n as f64 / mul_time.as_secs_f64() / 1_000_000.0;
        println!(
            "  Batch mul:  {:>8.3}ms  ({:.1} Mops/s)",
            mul_time.as_secs_f64() * 1000.0,
            mul_throughput
        );

        // Benchmark batch add
        let start = Instant::now();
        let _result = ctx.batch_add(&a, &b);
        let add_time = start.elapsed();
        let add_throughput = n as f64 / add_time.as_secs_f64() / 1_000_000.0;
        println!(
            "  Batch add:  {:>8.3}ms  ({:.1} Mops/s)",
            add_time.as_secs_f64() * 1000.0,
            add_throughput
        );

        // Benchmark parallel sum
        let start = Instant::now();
        let _sum = ctx.parallel_sum(&a);
        let sum_time = start.elapsed();
        let sum_throughput = n as f64 / sum_time.as_secs_f64() / 1_000_000.0;
        println!(
            "  Par sum:    {:>8.3}ms  ({:.1} Mops/s)",
            sum_time.as_secs_f64() * 1000.0,
            sum_throughput
        );

        // CPU comparison with real field arithmetic
        println!("\n  CPU comparison (arkworks BN254 Fr):");

        use ark_bn254::Fr;
        use ark_ff::{BigInteger, Field, PrimeField};
        use ark_std::UniformRand;
        let mut rng = ark_std::test_rng();

        let cpu_a: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let cpu_b: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

        // CPU batch multiply (single-threaded)
        let start = Instant::now();
        let cpu_result: Vec<Fr> = cpu_a
            .iter()
            .zip(cpu_b.iter())
            .map(|(a, b)| *a * *b)
            .collect();
        let cpu_mul_time = start.elapsed();
        let cpu_mul_throughput = n as f64 / cpu_mul_time.as_secs_f64() / 1_000_000.0;
        println!(
            "  CPU mul:    {:>8.3}ms  ({:.1} Mops/s)",
            cpu_mul_time.as_secs_f64() * 1000.0,
            cpu_mul_throughput
        );

        // CPU batch multiply (rayon parallel)
        use rayon::prelude::*;
        let start = Instant::now();
        let _cpu_par_result: Vec<Fr> = cpu_a
            .par_iter()
            .zip(cpu_b.par_iter())
            .map(|(a, b)| *a * *b)
            .collect();
        let cpu_par_mul_time = start.elapsed();
        let cpu_par_throughput = n as f64 / cpu_par_mul_time.as_secs_f64() / 1_000_000.0;
        println!(
            "  CPU par:    {:>8.3}ms  ({:.1} Mops/s, {} threads)",
            cpu_par_mul_time.as_secs_f64() * 1000.0,
            cpu_par_throughput,
            rayon::current_num_threads()
        );

        // Extract internal Montgomery-form representation.
        // arkworks Fr is repr-transparent over BigInt<4>, which is [u64; 4].
        // into_bigint() converts OUT of Montgomery form. We need the RAW form.
        fn fr_to_mont_limbs(f: &Fr) -> [u64; 4] {
            // Fr stores value as aR mod p internally.
            // We can get it by: f * R^2 in standard, or just read the raw bytes.
            // Safest: use transmute since Fp is repr-transparent over BigInt<4>.
            unsafe { std::mem::transmute::<Fr, [u64; 4]>(*f) }
        }

        // GPU mul using Montgomery-form data
        let gpu_a: Vec<GpuFp256> = cpu_a
            .iter()
            .map(|f| GpuFp256::from_u64_limbs(&fr_to_mont_limbs(f)))
            .collect();
        let gpu_b: Vec<GpuFp256> = cpu_b
            .iter()
            .map(|f| GpuFp256::from_u64_limbs(&fr_to_mont_limbs(f)))
            .collect();

        let start = Instant::now();
        let gpu_result = ctx.batch_mul(&gpu_a, &gpu_b);
        let gpu_real_time = start.elapsed();
        let gpu_real_throughput = n as f64 / gpu_real_time.as_secs_f64() / 1_000_000.0;
        println!(
            "  GPU mul:    {:>8.3}ms  ({:.1} Mops/s)",
            gpu_real_time.as_secs_f64() * 1000.0,
            gpu_real_throughput
        );

        // Correctness check: compare GPU Montgomery-form output with CPU Montgomery-form result
        let check_n = 10.min(n);
        let mut correct = 0;
        for i in 0..check_n {
            let expected = fr_to_mont_limbs(&cpu_result[i]);
            let got = gpu_result[i].to_u64_limbs();
            if expected == got {
                correct += 1;
            } else if i < 3 {
                println!("\n  Mismatch at [{i}]:");
                println!(
                    "    CPU mont = {:016x} {:016x} {:016x} {:016x}",
                    expected[3], expected[2], expected[1], expected[0]
                );
                println!(
                    "    GPU mont = {:016x} {:016x} {:016x} {:016x}",
                    got[3], got[2], got[1], got[0]
                );
            }
        }
        println!("\n  Correctness: {correct}/{check_n} match");

        println!("\n  ---- Speedup Summary ----");
        let vs_single = cpu_mul_time.as_secs_f64() / gpu_real_time.as_secs_f64();
        let vs_parallel = cpu_par_mul_time.as_secs_f64() / gpu_real_time.as_secs_f64();
        println!("  GPU vs CPU single:   {vs_single:.2}x");
        println!("  GPU vs CPU parallel: {vs_parallel:.2}x");

        Ok(())
    }
}

#[cfg(not(target_os = "macos"))]
pub mod gpu {
    pub fn benchmark_gpu(_n: usize) -> Result<(), String> {
        Err("Metal GPU is only available on macOS".to_string())
    }
}

/// Metal GPU-accelerated MSM (Multi-Scalar Multiplication) using Pippenger's bucket method.
#[cfg(target_os = "macos")]
pub mod msm {
    use metal::*;
    use std::mem;
    use std::time::Instant;

    const MSM_SHADER_SOURCE: &str = include_str!("../shaders/bn254_msm.metal");

    /// G1 affine point in GPU layout — must match G1Affine in bn254_msm.metal
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct GpuG1Affine {
        pub x: [u32; 8], // Fq256 in Montgomery form
        pub y: [u32; 8], // Fq256 in Montgomery form
        pub is_infinity: u32,
        pub _pad: [u32; 3],
    }

    /// G1 projective point in GPU layout — must match G1Projective in bn254_msm.metal
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct GpuG1Projective {
        pub x: [u32; 8],
        pub y: [u32; 8],
        pub z: [u32; 8],
    }

    /// Scatter entry — (bucket_idx, point_idx) pair
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct ScatterEntry {
        pub bucket_idx: u32,
        pub point_idx: u32,
    }

    impl GpuG1Affine {
        pub fn infinity() -> Self {
            Self {
                x: [0; 8],
                y: [0; 8],
                is_infinity: 1,
                _pad: [0; 3],
            }
        }

        /// Create from arkworks G1Affine point.
        /// Coordinates must be in Montgomery form (raw internal representation).
        pub fn from_ark(p: &ark_bn254::G1Affine) -> Self {
            use ark_ec::AffineRepr;
            if p.is_zero() {
                return Self::infinity();
            }
            let px = p.x().unwrap();
            let py = p.y().unwrap();
            let x = fq_to_mont_u32(&px);
            let y = fq_to_mont_u32(&py);
            Self {
                x,
                y,
                is_infinity: 0,
                _pad: [0; 3],
            }
        }
    }

    impl GpuG1Projective {
        pub fn identity() -> Self {
            Self {
                x: [0; 8],
                y: [0; 8],
                z: [0; 8],
            }
        }

        /// Convert to arkworks G1Projective
        pub fn to_ark(&self) -> ark_bn254::G1Projective {
            let x = fq_from_mont_u32(&self.x);
            let y = fq_from_mont_u32(&self.y);
            let z = fq_from_mont_u32(&self.z);
            ark_bn254::G1Projective::new_unchecked(x, y, z)
        }
    }

    /// Extract Montgomery-form u32 limbs from an arkworks Fq element
    fn fq_to_mont_u32(f: &ark_bn254::Fq) -> [u32; 8] {
        // Fq is repr-transparent over BigInt<4> which is [u64; 4]
        let u64_limbs: [u64; 4] = unsafe { std::mem::transmute::<ark_bn254::Fq, [u64; 4]>(*f) };
        [
            u64_limbs[0] as u32,
            (u64_limbs[0] >> 32) as u32,
            u64_limbs[1] as u32,
            (u64_limbs[1] >> 32) as u32,
            u64_limbs[2] as u32,
            (u64_limbs[2] >> 32) as u32,
            u64_limbs[3] as u32,
            (u64_limbs[3] >> 32) as u32,
        ]
    }

    /// Reconstruct an arkworks Fq from Montgomery-form u32 limbs
    fn fq_from_mont_u32(limbs: &[u32; 8]) -> ark_bn254::Fq {
        let u64_limbs: [u64; 4] = [
            limbs[0] as u64 | (limbs[1] as u64) << 32,
            limbs[2] as u64 | (limbs[3] as u64) << 32,
            limbs[4] as u64 | (limbs[5] as u64) << 32,
            limbs[6] as u64 | (limbs[7] as u64) << 32,
        ];
        unsafe { std::mem::transmute::<[u64; 4], ark_bn254::Fq>(u64_limbs) }
    }

    /// Extract scalar as u32 limbs (standard form, not Montgomery)
    fn fr_to_u32_limbs(f: &ark_bn254::Fr) -> [u32; 8] {
        use ark_ff::{BigInteger, PrimeField};
        let bigint = f.into_bigint();
        let u64s = bigint.0;
        [
            u64s[0] as u32,
            (u64s[0] >> 32) as u32,
            u64s[1] as u32,
            (u64s[1] >> 32) as u32,
            u64s[2] as u32,
            (u64s[2] >> 32) as u32,
            u64s[3] as u32,
            (u64s[3] >> 32) as u32,
        ]
    }

    /// Metal MSM context — compiles and holds the Pippenger shader pipelines
    pub struct MetalMsm {
        device: Device,
        queue: CommandQueue,
        pipeline_scatter: ComputePipelineState,
        pipeline_accumulate: ComputePipelineState,
    }

    impl MetalMsm {
        pub fn new() -> Result<Self, String> {
            let device = Device::system_default().ok_or("No Metal GPU device found")?;
            let queue = device.new_command_queue();

            let opts = CompileOptions::new();
            let library = device
                .new_library_with_source(MSM_SHADER_SOURCE, &opts)
                .map_err(|e| format!("MSM shader compile error: {e}"))?;

            let make_pipeline = |name: &str| -> Result<ComputePipelineState, String> {
                let func = library
                    .get_function(name, None)
                    .map_err(|e| format!("Function '{name}' not found: {e}"))?;
                device
                    .new_compute_pipeline_state_with_function(&func)
                    .map_err(|e| format!("Pipeline '{name}' error: {e}"))
            };

            let pipeline_scatter = make_pipeline("pippenger_scatter")?;
            let pipeline_accumulate = make_pipeline("pippenger_accumulate")?;

            Ok(Self {
                device,
                queue,
                pipeline_scatter,
                pipeline_accumulate,
            })
        }

        /// Compute MSM: sum_i scalars[i] * points[i]
        /// Uses Pippenger's bucket method with GPU-accelerated scatter and accumulate phases.
        pub fn msm(
            &self,
            points: &[GpuG1Affine],
            scalars: &[ark_bn254::Fr],
        ) -> Result<GpuG1Projective, String> {
            let n = points.len();
            assert_eq!(n, scalars.len());
            if n == 0 {
                return Ok(GpuG1Projective::identity());
            }

            // Choose window size c based on n (heuristic from gnark/bellman)
            let c = optimal_window_bits(n);
            let num_buckets = 1u32 << c;
            let num_windows = (256 + c - 1) / c;

            // Convert scalars to u32 limbs (standard form for bit extraction)
            let scalar_u32s: Vec<u32> = scalars.iter().flat_map(|s| fr_to_u32_limbs(s)).collect();

            // Upload points and scalars to GPU once
            let buf_points = self.device.new_buffer_with_data(
                points.as_ptr() as *const _,
                (n * mem::size_of::<GpuG1Affine>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let buf_scalars = self.device.new_buffer_with_data(
                scalar_u32s.as_ptr() as *const _,
                (scalar_u32s.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Process each window
            let mut window_sums: Vec<GpuG1Projective> = Vec::with_capacity(num_windows);

            let t_windows = Instant::now();
            for w in 0..num_windows {
                let window_sum = self.process_window(
                    &buf_points,
                    &buf_scalars,
                    n,
                    c as u32,
                    w as u32,
                    num_buckets,
                )?;
                window_sums.push(window_sum);
            }
            let windows_elapsed = t_windows.elapsed();

            let t_combine = Instant::now();
            let result = combine_windows(&window_sums, c);
            let combine_elapsed = t_combine.elapsed();

            if n >= 1024 {
                println!("    MSM breakdown: c={c}, windows={num_windows}, buckets={num_buckets}");
                println!(
                    "    Window processing: {:>8.3}ms",
                    windows_elapsed.as_secs_f64() * 1000.0
                );
                println!(
                    "    Window combining:  {:>8.3}ms",
                    combine_elapsed.as_secs_f64() * 1000.0
                );
            }

            Ok(result)
        }

        /// Process a single Pippenger window on GPU
        fn process_window(
            &self,
            buf_points: &Buffer,
            buf_scalars: &Buffer,
            n: usize,
            window_bits: u32,
            window_idx: u32,
            num_buckets: u32,
        ) -> Result<GpuG1Projective, String> {
            // Phase 1: Scatter — assign each point to a bucket
            let buf_scatter = self.device.new_buffer(
                (n * mem::size_of::<ScatterEntry>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let params = [n as u32, window_bits, window_idx];
            let buf_params = self.device.new_buffer_with_data(
                params.as_ptr() as *const _,
                12,
                MTLResourceOptions::StorageModeShared,
            );

            let cmd = self.queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline_scatter);
            encoder.set_buffer(0, Some(buf_scalars), 0);
            encoder.set_buffer(1, Some(&buf_scatter), 0);
            encoder.set_buffer(2, Some(&buf_params), 0);

            let grid = MTLSize::new(n as u64, 1, 1);
            let group_size =
                (self.pipeline_scatter.max_total_threads_per_threadgroup() as u64).min(n as u64);
            let group = MTLSize::new(group_size, 1, 1);
            encoder.dispatch_threads(grid, group);
            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // Read scatter results back to CPU for sorting
            let scatter_ptr = buf_scatter.contents() as *const ScatterEntry;
            let mut scatter: Vec<ScatterEntry> = vec![ScatterEntry::default(); n];
            unsafe {
                std::ptr::copy_nonoverlapping(scatter_ptr, scatter.as_mut_ptr(), n);
            }

            // Sort scatter entries by bucket index (CPU — radix sort would be ideal for GPU)
            scatter.sort_unstable_by_key(|e| e.bucket_idx);

            // Compute bucket offsets (prefix sum)
            let mut bucket_offsets = vec![0u32; (num_buckets + 1) as usize];
            {
                let mut current_bucket = 0u32;
                for (i, entry) in scatter.iter().enumerate() {
                    while current_bucket < entry.bucket_idx {
                        current_bucket += 1;
                        bucket_offsets[current_bucket as usize] = i as u32;
                    }
                }
                // Fill remaining buckets
                while current_bucket < num_buckets {
                    current_bucket += 1;
                    bucket_offsets[current_bucket as usize] = n as u32;
                }
            }

            // Phase 2: Accumulate — GPU accumulates points within each bucket
            let buf_sorted_scatter = self.device.new_buffer_with_data(
                scatter.as_ptr() as *const _,
                (n * mem::size_of::<ScatterEntry>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let buf_offsets = self.device.new_buffer_with_data(
                bucket_offsets.as_ptr() as *const _,
                (bucket_offsets.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let buf_bucket_sums = self.device.new_buffer(
                (num_buckets as u64) * mem::size_of::<GpuG1Projective>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let cmd2 = self.queue.new_command_buffer();
            let encoder2 = cmd2.new_compute_command_encoder();
            encoder2.set_compute_pipeline_state(&self.pipeline_accumulate);
            encoder2.set_buffer(0, Some(buf_points), 0);
            encoder2.set_buffer(1, Some(&buf_sorted_scatter), 0);
            encoder2.set_buffer(2, Some(&buf_offsets), 0);
            encoder2.set_buffer(3, Some(&buf_bucket_sums), 0);

            // Skip bucket 0 (identity scalar) — start from bucket 1
            let active_buckets = num_buckets; // includes bucket 0 for simplicity
            let grid2 = MTLSize::new(active_buckets as u64, 1, 1);
            let group_size2 = (self.pipeline_accumulate.max_total_threads_per_threadgroup() as u64)
                .min(active_buckets as u64);
            let group2 = MTLSize::new(group_size2, 1, 1);
            encoder2.dispatch_threads(grid2, group2);
            encoder2.end_encoding();
            cmd2.commit();
            cmd2.wait_until_completed();

            // Read bucket sums
            let bucket_ptr = buf_bucket_sums.contents() as *const GpuG1Projective;
            let mut bucket_sums = vec![GpuG1Projective::identity(); num_buckets as usize];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bucket_ptr,
                    bucket_sums.as_mut_ptr(),
                    num_buckets as usize,
                );
            }

            // Reduce buckets for this window on CPU:
            // window_sum = sum_{i=1}^{2^c-1} i * bucket_sums[i]
            // Efficient: running_sum accumulates from top bucket down
            Ok(reduce_buckets(&bucket_sums))
        }
    }

    /// Reduce bucket sums into a single window result.
    /// Uses the running-sum trick: start from the top bucket, accumulate downward.
    /// This computes sum_{i=1}^{B-1} i * bucket[i] with B-1 additions instead of naive B*c doublings.
    fn reduce_buckets(bucket_sums: &[GpuG1Projective]) -> GpuG1Projective {
        use ark_bn254::G1Projective as ArkG1;
        use std::ops::AddAssign;

        let mut running = ArkG1::default();
        let mut window_sum = ArkG1::default();

        for i in (1..bucket_sums.len()).rev() {
            let bucket_ark = bucket_sums[i].to_ark();
            running.add_assign(&bucket_ark);
            window_sum.add_assign(&running);
        }

        ark_projective_to_gpu(&window_sum)
    }

    /// Combine window results: result = sum_w window_sums[w] << (w * c)
    /// where << means repeated doubling in the EC group.
    fn combine_windows(window_sums: &[GpuG1Projective], c: usize) -> GpuG1Projective {
        use ark_bn254::G1Projective as ArkG1;
        use ark_ec::AdditiveGroup;
        use std::ops::AddAssign;

        let mut total = ArkG1::default();

        for w in (0..window_sums.len()).rev() {
            if w < window_sums.len() - 1 {
                for _ in 0..c {
                    total = total.double();
                }
            }
            let ws = window_sums[w].to_ark();
            total.add_assign(&ws);
        }

        ark_projective_to_gpu(&total)
    }

    /// Convert an arkworks G1Projective to our GPU representation
    fn ark_projective_to_gpu(p: &ark_bn254::G1Projective) -> GpuG1Projective {
        use ark_ec::AffineRepr;
        let aff: ark_bn254::G1Affine = (*p).into();
        if AffineRepr::is_zero(&aff) {
            GpuG1Projective::identity()
        } else {
            let px = aff.x().unwrap();
            let py = aff.y().unwrap();
            GpuG1Projective {
                x: fq_to_mont_u32(&px),
                y: fq_to_mont_u32(&py),
                z: [
                    0xc58f0d9d, 0xd35d438d, 0xf5c70b3d, 0x0a78eb28, 0x7879462c, 0x666ea36f,
                    0x9a07df2f, 0x0e0a77c1,
                ],
            }
        }
    }

    /// Choose optimal window bits for Pippenger based on number of points
    fn optimal_window_bits(n: usize) -> usize {
        if n < 32 {
            return 1;
        }
        let log_n = (n as f64).log2() as usize;
        // Heuristic: c ≈ log2(n) / 2, clamped to [4, 16]
        log_n.max(4).min(16) / 2 * 2 // round to even
    }

    /// Run MSM benchmark comparing GPU Pippenger vs CPU arkworks MSM
    pub fn benchmark_msm(n: usize) -> Result<(), String> {
        use ark_bn254::{Fr, G1Affine, G1Projective as ArkG1};
        use ark_ec::VariableBaseMSM;
        use ark_std::UniformRand;

        println!("\nMetal GPU MSM Benchmark (n={n})");
        println!("=================================");

        // Generate random test data
        let mut rng = ark_std::test_rng();
        let ark_scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let ark_points: Vec<G1Affine> = (0..n)
            .map(|_| {
                let p = ArkG1::rand(&mut rng);
                p.into()
            })
            .collect();

        // CPU MSM (arkworks)
        let start = Instant::now();
        let cpu_result = ArkG1::msm(&ark_points, &ark_scalars).unwrap();
        let cpu_time = start.elapsed();
        println!(
            "  CPU MSM (arkworks):  {:>8.3}ms",
            cpu_time.as_secs_f64() * 1000.0
        );

        // GPU MSM
        let ctx = MetalMsm::new()?;

        // Convert points to GPU format
        let gpu_points: Vec<GpuG1Affine> = ark_points
            .iter()
            .map(|p| GpuG1Affine::from_ark(p))
            .collect();

        // Warmup
        if n >= 64 {
            let _ = ctx.msm(&gpu_points[..64], &ark_scalars[..64]);
        }

        let start = Instant::now();
        let gpu_result = ctx.msm(&gpu_points, &ark_scalars)?;
        let gpu_time = start.elapsed();
        println!(
            "  GPU MSM (Metal):     {:>8.3}ms",
            gpu_time.as_secs_f64() * 1000.0
        );

        // Correctness check
        let gpu_ark = gpu_result.to_ark();
        let match_ok = gpu_ark == cpu_result;
        println!("  Correctness: {}", if match_ok { "PASS" } else { "FAIL" });

        if !match_ok {
            let cpu_aff: ark_bn254::G1Affine = cpu_result.into();
            let gpu_aff: ark_bn254::G1Affine = gpu_ark.into();
            println!("  CPU: {:?}", cpu_aff);
            println!("  GPU: {:?}", gpu_aff);
        }

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!("  Speedup: {speedup:.2}x");

        Ok(())
    }
}

#[cfg(not(target_os = "macos"))]
pub mod msm {
    pub fn benchmark_msm(_n: usize) -> Result<(), String> {
        Err("Metal GPU MSM is only available on macOS".to_string())
    }
}
