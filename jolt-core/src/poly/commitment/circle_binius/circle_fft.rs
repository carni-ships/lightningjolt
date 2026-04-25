//! Circle FFT implementation for CircleBinius backend
//!
//! This module implements FFT over the Circle group, which is used in
//! Circle STARKs for efficient proof generation.

use crate::poly::commitment::circle_binius::fields::{BinaryField32b, BinaryField64b};

/// Circle group element represented as a pair of field elements
#[derive(Debug, Clone, Copy)]
pub struct CirclePoint<F> {
    pub x: F,
    pub y: F,
}

impl CirclePoint<BinaryField32b> {
    /// Create a new circle point
    pub fn new(x: BinaryField32b, y: BinaryField32b) -> Self {
        Self { x, y }
    }

    /// Perform FFT of size n on the given evaluations
    pub fn fft(evaluations: &[BinaryField32b], _log_n: usize) -> Vec<BinaryField32b> {
        // TODO: Implement proper circle FFT
        // For now, just return a copy
        evaluations.to_vec()
    }
}

impl CirclePoint<BinaryField64b> {
    /// Create a new circle point
    pub fn new(x: BinaryField64b, y: BinaryField64b) -> Self {
        Self { x, y }
    }
}
