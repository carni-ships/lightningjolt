//! Metal GPU device management and shader compilation
//!
//! Caches the command queue and all pipeline states in a singleton to avoid
//! per-dispatch allocation overhead (~2-3ms saved per call).
#![allow(dead_code)]

use metal::{
    CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;
use std::sync::OnceLock;

static GPU: OnceLock<MetalGpu> = OnceLock::new();

/// Holds the Metal device, compiled shader library, cached command queue,
/// and pre-built pipeline states for all kernels.
pub(crate) struct MetalGpu {
    pub device: Device,
    pub library: Library,
    command_queue: CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
}

// SAFETY: Metal device, library, command queue, and pipeline states are thread-safe.
// The Metal framework guarantees these objects are safe to share across threads.
// Command buffer creation and commit are the synchronization points.
unsafe impl Send for MetalGpu {}
unsafe impl Sync for MetalGpu {}

/// Maximum thread group size for the Miller loop kernel.
/// The Miller loop uses ~800 bytes of registers per thread (G2HomProjective + Fq12 + temps).
/// Using max_total_threads_per_threadgroup (typically 1024) causes massive register spilling.
/// 64 threads keeps register pressure manageable on Apple Silicon GPUs.
const MILLER_LOOP_THREAD_GROUP_SIZE: u64 = 64;

impl MetalGpu {
    fn new() -> Self {
        let device = Device::system_default().expect("No Metal GPU device found");

        let shader_source = include_str!("fq.metal");
        let options = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(shader_source, &options)
            .expect("Failed to compile Metal shader");

        let command_queue = device.new_command_queue();

        // Pre-build all pipeline states at init time
        let kernel_names: &[&str] = &[
            "fq_mul_kernel",
            "fq_mul_chain_kernel",
            "fq2_mul_kernel",
            "fq6_mul_kernel",
            "fq12_mul_kernel",
            "miller_loop_kernel",
            "fq12_sqr_chain_kernel",
        ];

        let mut pipelines = HashMap::new();
        for &name in kernel_names {
            let function = library
                .get_function(name, None)
                .unwrap_or_else(|_| panic!("{name} not found in shader"));
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .expect("Failed to create compute pipeline");
            pipelines.insert(name, pipeline);
        }

        Self {
            device,
            library,
            command_queue,
            pipelines,
        }
    }

    fn get_pipeline(&self, name: &str) -> &ComputePipelineState {
        self.pipelines
            .get(name)
            .unwrap_or_else(|| panic!("{name} pipeline not cached"))
    }
}

/// Get or initialize the global Metal GPU context.
pub(crate) fn get_gpu() -> &'static MetalGpu {
    GPU.get_or_init(MetalGpu::new)
}

/// Run a batch of Fq multiplications on the GPU.
pub(crate) fn gpu_fq_mul_batch(a: &[[u32; 8]], b: &[[u32; 8]]) -> Vec<[u32; 8]> {
    gpu_dispatch_binary::<[u32; 8]>(a, b, "fq_mul_kernel")
}

/// Run a chain of Fq multiply + repeated squaring on the GPU.
pub(crate) fn gpu_fq_mul_chain(a: &[[u32; 8]], b: &[[u32; 8]], iters: u32) -> Vec<[u32; 8]> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 {
        return vec![];
    }

    let gpu = get_gpu();
    let device = &gpu.device;
    let pipeline = gpu.get_pipeline("fq_mul_chain_kernel");

    let buf_size = std::mem::size_of_val(a) as u64;

    let buf_a = device.new_buffer_with_data(
        a.as_ptr() as *const _,
        buf_size,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_b = device.new_buffer_with_data(
        b.as_ptr() as *const _,
        buf_size,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_result = device.new_buffer(buf_size, MTLResourceOptions::StorageModeShared);
    let buf_iters = device.new_buffer_with_data(
        &iters as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = gpu.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_a), 0);
    encoder.set_buffer(1, Some(&buf_b), 0);
    encoder.set_buffer(2, Some(&buf_result), 0);
    encoder.set_buffer(3, Some(&buf_iters), 0);

    let thread_group_size = MTLSize::new(
        std::cmp::min(n as u64, pipeline.max_total_threads_per_threadgroup()),
        1,
        1,
    );
    let grid_size = MTLSize::new(n as u64, 1, 1);

    encoder.dispatch_threads(grid_size, thread_group_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result_ptr = buf_result.contents() as *const [u32; 8];
    let mut results = vec![[0u32; 8]; n];
    unsafe {
        std::ptr::copy_nonoverlapping(result_ptr, results.as_mut_ptr(), n);
    }
    results
}

pub(crate) type Fq2Limbs = [u32; 16];
pub(crate) type G1AffineLimbs = [u32; 16]; // x, y as Fq
pub(crate) type G2AffineLimbs = [u32; 32]; // x.c0, x.c1, y.c0, y.c1 as Fq
pub(crate) type Fq12Limbs = [u32; 96];

/// Run a batch of Fq2 multiplications on the GPU.
pub(crate) fn gpu_fq2_mul_batch(a: &[Fq2Limbs], b: &[Fq2Limbs]) -> Vec<Fq2Limbs> {
    gpu_dispatch_binary::<Fq2Limbs>(a, b, "fq2_mul_kernel")
}

/// Run a batch of Fq12 multiplications on the GPU.
pub(crate) fn gpu_fq12_mul_batch(a: &[Fq12Limbs], b: &[Fq12Limbs]) -> Vec<Fq12Limbs> {
    gpu_dispatch_binary::<Fq12Limbs>(a, b, "fq12_mul_kernel")
}

/// Run Miller loops on GPU. Each thread computes one independent (G1, G2) pairing.
/// Returns Fq12 results (before final exponentiation).
pub(crate) fn gpu_miller_loop(g1: &[G1AffineLimbs], g2: &[G2AffineLimbs]) -> Vec<Fq12Limbs> {
    assert_eq!(g1.len(), g2.len());
    let n = g1.len();
    if n == 0 {
        return vec![];
    }

    let gpu = get_gpu();
    let device = &gpu.device;
    let pipeline = gpu.get_pipeline("miller_loop_kernel");

    let g1_size = std::mem::size_of_val(g1) as u64;
    let g2_size = std::mem::size_of_val(g2) as u64;
    let result_size = (n * std::mem::size_of::<Fq12Limbs>()) as u64;

    let buf_g1 = device.new_buffer_with_data(
        g1.as_ptr() as *const _,
        g1_size,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_g2 = device.new_buffer_with_data(
        g2.as_ptr() as *const _,
        g2_size,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_result = device.new_buffer(result_size, MTLResourceOptions::StorageModeShared);

    let command_buffer = gpu.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_g1), 0);
    encoder.set_buffer(1, Some(&buf_g2), 0);
    encoder.set_buffer(2, Some(&buf_result), 0);

    // Use smaller thread group for Miller loop to reduce register spilling
    let thread_group_size = MTLSize::new(
        std::cmp::min(n as u64, MILLER_LOOP_THREAD_GROUP_SIZE),
        1,
        1,
    );
    let grid_size = MTLSize::new(n as u64, 1, 1);

    encoder.dispatch_threads(grid_size, thread_group_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result_ptr = buf_result.contents() as *const Fq12Limbs;
    let mut results: Vec<Fq12Limbs> = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(result_ptr, results.as_mut_ptr(), n);
        results.set_len(n);
    }
    results
}

/// Run a chain of Fq12 squarings on the GPU.
pub(crate) fn gpu_fq12_sqr_chain(input: &[Fq12Limbs], iters: u32) -> Vec<Fq12Limbs> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }

    let gpu = get_gpu();
    let device = &gpu.device;
    let pipeline = gpu.get_pipeline("fq12_sqr_chain_kernel");
    let buf_size = std::mem::size_of_val(input) as u64;

    let buf_input = device.new_buffer_with_data(
        input.as_ptr() as *const _,
        buf_size,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_result = device.new_buffer(buf_size, MTLResourceOptions::StorageModeShared);
    let buf_iters = device.new_buffer_with_data(
        &iters as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = gpu.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_input), 0);
    encoder.set_buffer(1, Some(&buf_result), 0);
    encoder.set_buffer(2, Some(&buf_iters), 0);

    let thread_group_size = MTLSize::new(
        std::cmp::min(n as u64, pipeline.max_total_threads_per_threadgroup()),
        1,
        1,
    );
    let grid_size = MTLSize::new(n as u64, 1, 1);

    encoder.dispatch_threads(grid_size, thread_group_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result_ptr = buf_result.contents() as *const Fq12Limbs;
    let mut results: Vec<Fq12Limbs> = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(result_ptr, results.as_mut_ptr(), n);
        results.set_len(n);
    }
    results
}

/// Generic GPU dispatch for batch binary operations (a[i] ⊗ b[i]).
/// Uses cached command queue and pipeline state.
fn gpu_dispatch_binary<T: Copy>(a: &[T], b: &[T], kernel_name: &str) -> Vec<T> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 {
        return vec![];
    }

    let gpu = get_gpu();
    let device = &gpu.device;
    let pipeline = gpu.get_pipeline(kernel_name);
    let buf_size = std::mem::size_of_val(a) as u64;

    let buf_a = device.new_buffer_with_data(
        a.as_ptr() as *const _,
        buf_size,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_b = device.new_buffer_with_data(
        b.as_ptr() as *const _,
        buf_size,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_result = device.new_buffer(buf_size, MTLResourceOptions::StorageModeShared);

    let command_buffer = gpu.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_a), 0);
    encoder.set_buffer(1, Some(&buf_b), 0);
    encoder.set_buffer(2, Some(&buf_result), 0);

    let thread_group_size = MTLSize::new(
        std::cmp::min(n as u64, pipeline.max_total_threads_per_threadgroup()),
        1,
        1,
    );
    let grid_size = MTLSize::new(n as u64, 1, 1);

    encoder.dispatch_threads(grid_size, thread_group_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result_ptr = buf_result.contents() as *const T;
    let mut results: Vec<T> = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(result_ptr, results.as_mut_ptr(), n);
        results.set_len(n);
    }
    results
}
