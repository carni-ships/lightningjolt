//! NTT Backend for Lattice-Based Dory
//!
//! This module provides the Number Theoretic Transform (NTT) implementation
//! using CPU-based algorithms. GPU acceleration via zkMetal is available
//! in the jolt-core lattice module.
//!
//! The NTT enables polynomial arithmetic in the frequency domain, where:
//! - Polynomial multiplication = pointwise multiplication in frequency domain
//! - Inner products = sum of pointwise products
//! - Fully parallelizable and GPU-friendly

pub mod cpu_ntt;
pub mod fp128;
pub mod gpu_ntt;
pub mod kyber;
pub mod poly_commit;

pub use fp128::{Fp128Element, Fp128Field};
pub use kyber::KyberCurve;
pub use cpu_ntt::TwiddleCache;

/// Maximum supported NTT size (2^26 = ~67M elements)
pub const MAX_NTT_SIZE: usize = 1 << 26;

/// NTT engine configuration
#[derive(Clone, Debug)]
pub struct NTTConfig {
    /// Log2 of the transform size
    pub log_size: usize,
    /// Whether to use GPU acceleration (Metal)
    pub use_gpu: bool,
    /// Inverse transform flag
    pub inverse: bool,
}

impl Default for NTTConfig {
    fn default() -> Self {
        Self {
            log_size: 12, // Default 4096 elements
            use_gpu: false,
            inverse: false,
        }
    }
}

impl NTTConfig {
    pub fn new(log_size: usize) -> Self {
        assert!(log_size <= 26, "Maximum NTT size is 2^26");
        Self {
            log_size,
            use_gpu: false,
            inverse: false,
        }
    }

    pub fn with_inverse(mut self, inverse: bool) -> Self {
        self.inverse = inverse;
        self
    }

    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }
}

/// Result of an NTT transform
#[derive(Clone, Debug)]
pub struct NTTResult<F> {
    pub values: Vec<F>,
    pub config: NTTConfig,
}

impl<F> NTTResult<F> {
    pub fn new(values: Vec<F>, config: NTTConfig) -> Self {
        Self { values, config }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Trait for NTT-capable fields
pub trait NTTField: Clone + Send + Sync {
    type Element: Clone + Send + Sync;

    /// Perform forward NTT transform
    fn forward(&self, input: &[Self::Element]) -> Vec<Self::Element>;

    /// Perform inverse NTT transform
    fn inverse(&self, input: &[Self::Element]) -> Vec<Self::Element>;

    /// Pointwise multiplication in frequency domain
    fn multiply(&self, a: &[Self::Element], b: &[Self::Element]) -> Vec<Self::Element>;

    /// Create NTT of polynomial coefficients
    fn poly_to_ntt(&self, poly: &[Self::Element]) -> Vec<Self::Element>;

    /// Convert from frequency domain back to coefficients
    fn ntt_to_poly(&self, ntt: &[Self::Element]) -> Vec<Self::Element>;
}
