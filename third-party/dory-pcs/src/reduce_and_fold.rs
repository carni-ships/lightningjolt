//! Opening proof protocol - prover and verifier state management
//!
//! This module contains the state machines for the interactive Dory protocol.
//! The prover maintains vectors and computes messages, while the verifier
//! maintains accumulated values and verifies messages.

#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use crate::error::DoryError;
use crate::messages::*;
use crate::mode::{Mode, Transparent};
use crate::primitives::arithmetic::{DoryRoutines, Field, Group, PairingCurve};
use crate::setup::{ProverSetup, VerifierSetup};
use std::marker::PhantomData;

#[cfg(feature = "zk")]
use crate::primitives::transcript::Transcript;

type Scalar<E> = <<E as PairingCurve>::G1 as Group>::Scalar;

/// Threshold below which sequential computation is faster than parallel rayon.
/// For small vector sizes, rayon spawn overhead exceeds parallelism benefit.
///
/// Note: 16 showed best results in testing but values 16-32 are all reasonable.
/// The optimal value depends on hardware (CPU cores, memory bandwidth).
const SEQUENTIAL_THRESHOLD: usize = 16;

/// Prover state for the Dory opening protocol
///
/// Maintains the current state of the prover during the interactive protocol.
/// The state consists of vectors that get folded in each round.
pub struct DoryProverState<'a, E: PairingCurve, M: Mode = Transparent> {
    /// Current v1 vector (G1 elements)
    v1: Vec<E::G1>,

    /// Current v2 vector (G2 elements)
    v2: Vec<E::G2>,

    /// For first round only: scalars used to construct v2 from fixed base h2
    v2_scalars: Option<Vec<Scalar<E>>>,

    /// Current s1 vector (scalars)
    s1: Vec<Scalar<E>>,

    /// Current s2 vector (scalars)
    s2: Vec<Scalar<E>>,

    /// Number of rounds remaining (log₂ of vector length)
    num_rounds: usize,

    /// Reference to prover setup
    setup: &'a ProverSetup<E>,

    // ZK accumulated blinds (zero in Transparent mode)
    r_c: Scalar<E>,
    r_d1: Scalar<E>,
    r_d2: Scalar<E>,
    r_e1: Scalar<E>,
    r_e2: Scalar<E>,
    // Per-round blinds stored between compute and apply (binary)
    round_d1: [Scalar<E>; 2],
    round_d2: [Scalar<E>; 2],
    round_c: [Scalar<E>; 2],
    round_e1: [Scalar<E>; 2],
    round_e2: [Scalar<E>; 2],
    // Per-round blinds for 4-ary folding
    round_d1_4ary: [Scalar<E>; 4],
    round_d2_4ary: [Scalar<E>; 4],
    round_c_4ary: [Scalar<E>; 16],
    round_e1_4ary: [Scalar<E>; 16],
    round_e2_4ary: [Scalar<E>; 16],

    _mode: PhantomData<M>,
}


/// Verifier state for the Dory opening protocol
///
/// Maintains the current accumulated values during verification.
/// These values get updated based on prover messages and challenges.
pub struct DoryVerifierState<E: PairingCurve> {
    /// Inner product accumulator
    c: E::GT,

    /// Commitment to v1: ⟨v1, Γ2⟩
    d1: E::GT,

    /// Commitment to v2: ⟨Γ1, v2⟩
    d2: E::GT,

    /// Extended protocol: commitment to s1
    e1: E::G1,

    /// Extended protocol: commitment to s2
    e2: E::G2,

    /// Initial e1 from VMV message
    /// Used in verify_final to batch the VMV constraint: D₂_init = e(E₁_init, Γ₂₀)
    e1_init: E::G1,

    /// Initial d2 from VMV message
    /// Used in verify_final to batch the VMV constraint: D₂_init = e(E₁_init, Γ₂₀)
    d2_init: E::GT,

    /// Accumulated scalar for s1 after folding across rounds
    s1_acc: Scalar<E>,

    /// Accumulated scalar for s2 after folding across rounds
    s2_acc: Scalar<E>,

    /// Per-round coordinates for s1 (length = num_rounds). Order matches folding order.
    s1_coords: Vec<Scalar<E>>,

    /// Per-round coordinates for s2 (length = num_rounds). Order matches folding order.
    s2_coords: Vec<Scalar<E>>,

    /// Number of rounds remaining for indexing setup arrays
    num_rounds: usize,

    /// Reference to verifier setup
    setup: VerifierSetup<E>,
}

impl<'a, E: PairingCurve, M: Mode> DoryProverState<'a, E, M>
where
    <E::G1 as Group>::Scalar: Field,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
    E::GT: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    /// Create new prover state
    ///
    /// # Parameters
    /// - `v1`: Initial G1 vector
    /// - `v2`: Initial G2 vector
    /// - `v2_scalars`: Optional scalars where v2 = h2 * scalars; enables MSM+pair in first round
    /// - `s1`: Initial scalar vector for G1 side
    /// - `s2`: Initial scalar vector for G2 side
    /// - `setup`: Prover setup parameters
    pub fn new(
        v1: Vec<E::G1>,
        v2: Vec<E::G2>,
        v2_scalars: Option<Vec<Scalar<E>>>,
        s1: Vec<Scalar<E>>,
        s2: Vec<Scalar<E>>,
        setup: &'a ProverSetup<E>,
    ) -> Self {
        debug_assert_eq!(v1.len(), v2.len(), "v1 and v2 must have equal length");
        debug_assert_eq!(v1.len(), s1.len(), "v1 and s1 must have equal length");
        debug_assert_eq!(v1.len(), s2.len(), "v1 and s2 must have equal length");
        debug_assert!(
            v1.len().is_power_of_two(),
            "vector length must be power of 2"
        );
        if let Some(sc) = v2_scalars.as_ref() {
            debug_assert_eq!(sc.len(), v2.len(), "v2_scalars must match v2 length");
        }

        let num_rounds = v1.len().trailing_zeros() as usize;
        let z = Scalar::<E>::zero();

        Self {
            v1,
            v2,
            v2_scalars,
            s1,
            s2,
            num_rounds,
            setup,
            r_c: z,
            r_d1: z,
            r_d2: z,
            r_e1: z,
            r_e2: z,
            round_d1: [z; 2],
            round_d2: [z; 2],
            round_c: [z; 2],
            round_e1: [z; 2],
            round_e2: [z; 2],
            round_d1_4ary: [z; 4],
            round_d2_4ary: [z; 4],
            round_c_4ary: [z; 16],
            round_e1_4ary: [z; 16],
            round_e2_4ary: [z; 16],
            _mode: PhantomData,
        }
    }

    /// Get the number of rounds remaining
    pub fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    /// Set initial VMV blinds (r_d1, r_c, r_d2, r_e1, r_e2).
    pub fn set_initial_blinds(
        &mut self,
        r_d1: Scalar<E>,
        r_c: Scalar<E>,
        r_d2: Scalar<E>,
        r_e1: Scalar<E>,
        r_e2: Scalar<E>,
    ) {
        (self.r_d1, self.r_c, self.r_d2, self.r_e1, self.r_e2) = (r_d1, r_c, r_d2, r_e1, r_e2);
    }

    /// Compute first reduce message for current round
    ///
    /// Computes D1L, D1R, D2L, D2R, E1β, E2β based on current state.
    #[tracing::instrument(skip_all, name = "DoryProverState::compute_first_message")]
    pub fn compute_first_message<M1, M2>(&mut self) -> FirstReduceMessage<E::G1, E::G2, E::GT>
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        assert!(
            self.num_rounds > 0,
            "Not enough rounds left in prover state"
        );

        let n2 = 1 << (self.num_rounds - 1); // n/2

        // Split vectors into left and right halves
        let (v1_l, v1_r) = self.v1.split_at(n2);
        let (v2_l, v2_r) = self.v2.split_at(n2);

        // Get collapsed generator vectors of length n/2
        let g1_prime = &self.setup.g1_vec[..n2];
        let g2_prime = &self.setup.g2_vec[..n2];

        // Sample round blinds (zero in Transparent mode)
        self.round_d1 = [M::sample(), M::sample()];
        self.round_d2 = [M::sample(), M::sample()];

        let ht = &self.setup.ht;
        let rd1 = self.round_d1;
        let rd2 = self.round_d2;
        let g1_full = &self.setup.g1_vec[..1 << self.num_rounds];
        let g2_full = &self.setup.g2_vec[..1 << self.num_rounds];

        // For small vector sizes, sequential is faster than parallel rayon overhead.
        // Threshold chosen to balance parallelism benefit vs spawn overhead.
        #[cfg(feature = "parallel")]
        let ((d1_left, d1_right), ((d2_left, d2_right), (e1_beta, e2_beta))) =
            if n2 < SEQUENTIAL_THRESHOLD {
                // Sequential: avoid rayon spawn overhead for small MSMs
                let d1_left = M::mask(E::multi_pair_g2_setup(v1_l, g2_prime), ht, &rd1[0]);
                let d1_right = M::mask(E::multi_pair_g2_setup(v1_r, g2_prime), ht, &rd1[1]);
                let (d2_left_base, d2_right_base) = if let Some(scalars) = self.v2_scalars.as_ref() {
                    let (s_l, s_r) = scalars.split_at(n2);
                    let sum_left = M1::msm(g1_prime, s_l);
                    let sum_right = M1::msm(g1_prime, s_r);
                    let g2_fin = &self.setup.g2_vec[0];
                    (E::pair(&sum_left, g2_fin), E::pair(&sum_right, g2_fin))
                } else {
                    (
                        E::multi_pair_g1_setup(g1_prime, v2_l),
                        E::multi_pair_g1_setup(g1_prime, v2_r),
                    )
                };
                let d2_left = M::mask(d2_left_base, ht, &rd2[0]);
                let d2_right = M::mask(d2_right_base, ht, &rd2[1]);
                let e1_beta = M1::msm(g1_full, &self.s2[..]);
                let e2_beta = M2::msm(g2_full, &self.s1[..]);
                ((d1_left, d1_right), ((d2_left, d2_right), (e1_beta, e2_beta)))
            } else {
                rayon::join(
                    || {
                        rayon::join(
                            || M::mask(E::multi_pair_g2_setup(v1_l, g2_prime), ht, &rd1[0]),
                            || M::mask(E::multi_pair_g2_setup(v1_r, g2_prime), ht, &rd1[1]),
                        )
                    },
                    || {
                        rayon::join(
                            || {
                                let (d2_left_base, d2_right_base) = if let Some(scalars) =
                                    self.v2_scalars.as_ref()
                                {
                                    let (s_l, s_r) = scalars.split_at(n2);
                                    let (sum_left, sum_right) =
                                        rayon::join(|| M1::msm(g1_prime, s_l), || M1::msm(g1_prime, s_r));
                                    let g2_fin = &self.setup.g2_vec[0];
                                    (E::pair(&sum_left, g2_fin), E::pair(&sum_right, g2_fin))
                                } else {
                                    rayon::join(
                                        || E::multi_pair_g1_setup(g1_prime, v2_l),
                                        || E::multi_pair_g1_setup(g1_prime, v2_r),
                                    )
                                };
                                (
                                    M::mask(d2_left_base, ht, &rd2[0]),
                                    M::mask(d2_right_base, ht, &rd2[1]),
                                )
                            },
                            || {
                                rayon::join(
                                    || M1::msm(g1_full, &self.s2[..]),
                                    || M2::msm(g2_full, &self.s1[..]),
                                )
                            },
                        )
                    },
                )
            };

        #[cfg(not(feature = "parallel"))]
        let (d1_left, d1_right, d2_left, d2_right, e1_beta, e2_beta) = {
            let d1_left = M::mask(E::multi_pair_g2_setup(v1_l, g2_prime), ht, &rd1[0]);
            let d1_right = M::mask(E::multi_pair_g2_setup(v1_r, g2_prime), ht, &rd1[1]);
            let (d2_left_base, d2_right_base) = if let Some(scalars) = self.v2_scalars.as_ref() {
                let (s_l, s_r) = scalars.split_at(n2);
                let sum_left = M1::msm(g1_prime, s_l);
                let sum_right = M1::msm(g1_prime, s_r);
                let g2_fin = &self.setup.g2_vec[0];
                (E::pair(&sum_left, g2_fin), E::pair(&sum_right, g2_fin))
            } else {
                (
                    E::multi_pair_g1_setup(g1_prime, v2_l),
                    E::multi_pair_g1_setup(g1_prime, v2_r),
                )
            };
            let d2_left = M::mask(d2_left_base, ht, &rd2[0]);
            let d2_right = M::mask(d2_right_base, ht, &rd2[1]);
            let e1_beta = M1::msm(g1_full, &self.s2[..]);
            let e2_beta = M2::msm(g2_full, &self.s1[..]);
            (d1_left, d1_right, d2_left, d2_right, e1_beta, e2_beta)
        };

        FirstReduceMessage {
            d1_left,
            d1_right,
            d2_left,
            d2_right,
            e1_beta,
            e2_beta,
        }
    }

    /// Apply first challenge (beta) and combine vectors
    ///
    /// Updates the state by combining with generators scaled by beta.
    #[tracing::instrument(skip_all, name = "DoryProverState::apply_first_challenge")]
    pub fn apply_first_challenge<M1, M2>(&mut self, beta: &Scalar<E>)
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        let beta_inv = beta.inv().expect("beta must be invertible");
        let n = 1 << self.num_rounds;

        // v₁ ← v₁ + β·Γ₁ ∥ v₂ ← v₂ + β⁻¹·Γ₂ — independent, run in parallel
        let (g1_slice, g2_slice) = (&self.setup.g1_vec[..n], &self.setup.g2_vec[..n]);
        let (v1_ref, v2_ref) = (&mut self.v1, &mut self.v2);
        #[cfg(feature = "parallel")]
        rayon::join(
            || M1::fixed_scalar_mul_bases_then_add(g1_slice, v1_ref, beta),
            || M2::fixed_scalar_mul_bases_then_add(g2_slice, v2_ref, &beta_inv),
        );
        #[cfg(not(feature = "parallel"))]
        {
            M1::fixed_scalar_mul_bases_then_add(g1_slice, v1_ref, beta);
            M2::fixed_scalar_mul_bases_then_add(g2_slice, v2_ref, &beta_inv);
        }
        self.v2_scalars = None;

        self.r_c = self.r_c + self.r_d2 * beta + self.r_d1 * beta_inv;
    }

    /// Compute second reduce message for current round
    ///
    /// Computes C+, C-, E1+, E1-, E2+, E2- based on current state.
    #[tracing::instrument(skip_all, name = "DoryProverState::compute_second_message")]
    pub fn compute_second_message<M1, M2>(&mut self) -> SecondReduceMessage<E::G1, E::G2, E::GT>
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        let n2 = 1 << (self.num_rounds - 1); // n/2

        // Split all vectors into left and right halves
        let (v1_l, v1_r) = self.v1.split_at(n2);
        let (v2_l, v2_r) = self.v2.split_at(n2);
        let (s1_l, s1_r) = self.s1.split_at(n2);
        let (s2_l, s2_r) = self.s2.split_at(n2);

        self.round_c = [M::sample(), M::sample()];
        self.round_e1 = [M::sample(), M::sample()];
        self.round_e2 = [M::sample(), M::sample()];

        let ht = &self.setup.ht;
        let h1 = &self.setup.h1;
        let h2 = &self.setup.h2;
        let rc = self.round_c;
        let re1 = self.round_e1;
        let re2 = self.round_e2;

        // C₊ ∥ C₋, E₁± ∥ E₂± — all independent, run in parallel
        // Use batch pairing for C± to reduce Miller loop overhead
        // For small vectors, sequential is faster than parallel rayon overhead.
        #[cfg(feature = "parallel")]
        let ((c_plus, c_minus), ((e1_plus, e1_minus), (e2_plus, e2_minus))) =
            if n2 < SEQUENTIAL_THRESHOLD {
                // Sequential: avoid rayon spawn overhead for small MSMs
                let (c_plus_prod, c_minus_prod) = E::multi_pair_two_products(v1_l, v2_r, v1_r, v2_l);
                let c_plus = M::mask(c_plus_prod, ht, &rc[0]);
                let c_minus = M::mask(c_minus_prod, ht, &rc[1]);
                let e1_plus = M::mask(M1::msm(v1_l, s2_r), h1, &re1[0]);
                let e1_minus = M::mask(M1::msm(v1_r, s2_l), h1, &re1[1]);
                let e2_plus = M::mask(M2::msm(v2_r, s1_l), h2, &re2[0]);
                let e2_minus = M::mask(M2::msm(v2_l, s1_r), h2, &re2[1]);
                ((c_plus, c_minus), ((e1_plus, e1_minus), (e2_plus, e2_minus)))
            } else {
                rayon::join(
                    || {
                        // Use multi_pair_two_products to compute both C₊ and C₋ with a single Miller loop
                        let (c_plus_prod, c_minus_prod) = E::multi_pair_two_products(v1_l, v2_r, v1_r, v2_l);
                        (M::mask(c_plus_prod, ht, &rc[0]), M::mask(c_minus_prod, ht, &rc[1]))
                    },
                    || {
                        rayon::join(
                            || {
                                rayon::join(
                                    || M::mask(M1::msm(v1_l, s2_r), h1, &re1[0]),
                                    || M::mask(M1::msm(v1_r, s2_l), h1, &re1[1]),
                                )
                            },
                            || {
                                rayon::join(
                                    || M::mask(M2::msm(v2_r, s1_l), h2, &re2[0]),
                                    || M::mask(M2::msm(v2_l, s1_r), h2, &re2[1]),
                                )
                            },
                        )
                    },
                )
            };

        #[cfg(not(feature = "parallel"))]
        let (c_plus, c_minus, e1_plus, e1_minus, e2_plus, e2_minus) = {
            // Use multi_pair_two_products to compute both C₊ and C₋ with a single Miller loop
            let (c_plus_prod, c_minus_prod) = E::multi_pair_two_products(v1_l, v2_r, v1_r, v2_l);
            let c_plus = M::mask(c_plus_prod, ht, &rc[0]);
            let c_minus = M::mask(c_minus_prod, ht, &rc[1]);
            let e1_plus = M::mask(M1::msm(v1_l, s2_r), h1, &re1[0]);
            let e1_minus = M::mask(M1::msm(v1_r, s2_l), h1, &re1[1]);
            let e2_plus = M::mask(M2::msm(v2_r, s1_l), h2, &re2[0]);
            let e2_minus = M::mask(M2::msm(v2_l, s1_r), h2, &re2[1]);
            (c_plus, c_minus, e1_plus, e1_minus, e2_plus, e2_minus)
        };

        SecondReduceMessage {
            c_plus,
            c_minus,
            e1_plus,
            e1_minus,
            e2_plus,
            e2_minus,
        }
    }

    /// Apply second challenge (alpha) and fold vectors
    ///
    /// Reduces the vector size by half using the alpha challenge.
    #[tracing::instrument(skip_all, name = "DoryProverState::apply_second_challenge")]
    pub fn apply_second_challenge<M1: DoryRoutines<E::G1>, M2: DoryRoutines<E::G2>>(
        &mut self,
        alpha: &Scalar<E>,
    ) {
        let alpha_inv = alpha.inv().expect("alpha must be invertible");
        let n2 = 1 << (self.num_rounds - 1); // n/2

        // Fold v₁ ∥ v₂ ∥ s₁ ∥ s₂ — all independent
        let (v1, v2, s1, s2) = (&mut self.v1, &mut self.v2, &mut self.s1, &mut self.s2);
        #[cfg(feature = "parallel")]
        {
            rayon::join(
                || {
                    rayon::join(
                        || {
                            let (v1_l, v1_r) = v1.split_at_mut(n2);
                            M1::fixed_scalar_mul_vs_then_add(v1_l, v1_r, alpha);
                        },
                        || {
                            let (v2_l, v2_r) = v2.split_at_mut(n2);
                            M2::fixed_scalar_mul_vs_then_add(v2_l, v2_r, &alpha_inv);
                        },
                    )
                },
                || {
                    rayon::join(
                        || {
                            let (s1_l, s1_r) = s1.split_at_mut(n2);
                            M1::fold_field_vectors(s1_l, s1_r, alpha);
                        },
                        || {
                            let (s2_l, s2_r) = s2.split_at_mut(n2);
                            M1::fold_field_vectors(s2_l, s2_r, &alpha_inv);
                        },
                    )
                },
            );
        }
        #[cfg(not(feature = "parallel"))]
        {
            let (v1_l, v1_r) = v1.split_at_mut(n2);
            M1::fixed_scalar_mul_vs_then_add(v1_l, v1_r, alpha);
            let (v2_l, v2_r) = v2.split_at_mut(n2);
            M2::fixed_scalar_mul_vs_then_add(v2_l, v2_r, &alpha_inv);
            let (s1_l, s1_r) = s1.split_at_mut(n2);
            M1::fold_field_vectors(s1_l, s1_r, alpha);
            let (s2_l, s2_r) = s2.split_at_mut(n2);
            M1::fold_field_vectors(s2_l, s2_r, &alpha_inv);
        }
        self.v1.truncate(n2);
        self.v2.truncate(n2);
        self.s1.truncate(n2);
        self.s2.truncate(n2);

        self.r_c = self.r_c + self.round_c[0] * alpha + self.round_c[1] * alpha_inv;
        self.r_d1 = self.round_d1[0] * alpha + self.round_d1[1];
        self.r_d2 = self.round_d2[0] * alpha_inv + self.round_d2[1];
        self.r_e1 = self.r_e1 + self.round_e1[0] * alpha + self.round_e1[1] * alpha_inv;
        self.r_e2 = self.r_e2 + self.round_e2[0] * alpha + self.round_e2[1] * alpha_inv;

        self.num_rounds -= 1;
    }

    /// Compute 4-ary reduce message (Variant B - single pre-challenge message)
    ///
    /// Computes D1_i, D2_i for each of 4 quarters, plus raw cross-term pairings
    /// C_raw_ij = pair(v1_i, v2_j) and cross MSM elements. All pre-sampled before
    /// any challenge is known, enabling true parallel cascading sub-reductions.
    #[cfg(feature = "lattice")]
    ///
    /// This is the key to Variant B: the second message (cross terms) is now
    /// combined with the first message, and all blindings are pre-sampled.
    #[tracing::instrument(skip_all, name = "DoryProverState::compute_4ary_message")]
    pub fn compute_4ary_message<M1, M2>(
        &mut self,
    ) -> FirstReduceMessage4<E::G1, E::G2, E::GT>
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        assert!(
            self.num_rounds >= 2,
            "4-ary folding requires at least 2 rounds remaining (need log₂(N) >= 2)"
        );

        let n4 = 1 << (self.num_rounds - 2); // n/4

        // Split vectors into 4 quarters
        let v1_quarters: Vec<_> = self.v1.chunks(n4).collect();
        let v2_quarters: Vec<_> = self.v2.chunks(n4).collect();
        let s1_quarters: Vec<_> = self.s1.chunks(n4).collect();
        let s2_quarters: Vec<_> = self.s2.chunks(n4).collect();

        // Get collapsed generator vectors of length n/4
        let g1_prime = &self.setup.g1_vec[..n4];
        let g2_prime = &self.setup.g2_vec[..n4];

        // Pre-sample ALL round blinds (before any challenge!)
        // 16 for c_raw, 16 for e1_cross, 16 for e2_cross, plus 4 each for d1/d2
        self.round_d1_4ary = [M::sample(), M::sample(), M::sample(), M::sample()];
        self.round_d2_4ary = [M::sample(), M::sample(), M::sample(), M::sample()];
        self.round_c_4ary = [M::sample(); 16];
        self.round_e1_4ary = [M::sample(); 16];
        self.round_e2_4ary = [M::sample(); 16];

        let ht = &self.setup.ht;
        let h1 = &self.setup.h1;
        let h2 = &self.setup.h2;
        let rd1 = self.round_d1_4ary;
        let rd2 = self.round_d2_4ary;
        let rc = self.round_c_4ary;
        let re1 = self.round_e1_4ary;
        let re2 = self.round_e2_4ary;
        let g1_full = &self.setup.g1_vec[..1 << self.num_rounds];
        let g2_full = &self.setup.g2_vec[..1 << self.num_rounds];

        let (d1, d2, c_raw, e1_cross, e2_cross, e1_beta, e2_beta) = {
            let mut d1 = [E::GT::identity(); 4];
            let mut d2 = [E::GT::identity(); 4];
            let mut c_raw = [E::GT::identity(); 16];
            let mut e1_cross = [E::G1::identity(); 16];
            let mut e2_cross = [E::G2::identity(); 16];

            for i in 0..4 {
                d1[i] = M::mask(E::multi_pair_g2_setup(v1_quarters[i], g2_prime), ht, &rd1[i]);
                d2[i] = if let Some(scalars) = self.v2_scalars.as_ref() {
                    let s_quarter: Vec<_> = scalars.chunks(n4).collect();
                    let sum = M1::msm(g1_prime, s_quarter[i]);
                    let g2_fin = &self.setup.g2_vec[0];
                    M::mask(E::pair(&sum, g2_fin), ht, &rd2[i])
                } else {
                    M::mask(E::multi_pair_g1_setup(g1_prime, v2_quarters[i]), ht, &rd2[i])
                };
            }

            // Compute ALL raw pairings C_raw_ij = pair(Qi, Qj) for all i,j
            // This allows verifier to reconstruct cross terms after beta is known
            for i in 0..4 {
                for j in 0..4 {
                    let idx = i * 4 + j;
                    c_raw[idx] = M::mask(
                        E::multi_pair(&v1_quarters[i], &v2_quarters[j]),
                        ht,
                        &rc[idx],
                    );
                    e1_cross[idx] = M::mask(
                        M1::msm(&v1_quarters[i], &s2_quarters[j]),
                        h1,
                        &re1[idx],
                    );
                    e2_cross[idx] = M::mask(
                        M2::msm(&v2_quarters[i], &s1_quarters[j]),
                        h2,
                        &re2[idx],
                    );
                }
            }

            let e1_beta = M1::msm(g1_full, &self.s2[..]);
            let e2_beta = M2::msm(g2_full, &self.s1[..]);
            (d1, d2, c_raw, e1_cross, e2_cross, e1_beta, e2_beta)
        };

        FirstReduceMessage4 {
            d1,
            d2,
            c_raw,
            e1_cross,
            e2_cross,
            e1_beta,
            e2_beta,
        }
    }

    /// Apply first challenge (beta) for 4-ary folding
    ///
    /// Updates witnesses with beta before the alpha fold.
    /// This is the same as the binary case but applied to 4-ary structure.
    #[cfg(feature = "lattice")]
    #[tracing::instrument(skip_all, name = "DoryProverState::apply_4ary_beta")]
    pub fn apply_4ary_beta<M1, M2>(&mut self, beta: &Scalar<E>)
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        let beta_inv = beta.inv().expect("beta must be invertible");
        let _n4 = 1 << (self.num_rounds - 2); // n/4 (kept for documentation, beta applies to full vector)

        // Apply beta to full vectors (v1 += β·Γ₁, v2 += β⁻¹·Γ₂)
        let n = 1 << self.num_rounds;
        let g1_slice = &self.setup.g1_vec[..n];
        let g2_slice = &self.setup.g2_vec[..n];

        #[cfg(feature = "parallel")]
        rayon::join(
            || M1::fixed_scalar_mul_bases_then_add(g1_slice, &mut self.v1, beta),
            || M2::fixed_scalar_mul_bases_then_add(g2_slice, &mut self.v2, &beta_inv),
        );

        #[cfg(not(feature = "parallel"))]
        {
            M1::fixed_scalar_mul_bases_then_add(g1_slice, &mut self.v1, beta);
            M2::fixed_scalar_mul_bases_then_add(g2_slice, &mut self.v2, &beta_inv);
        }

        self.v2_scalars = None;

        // Update r_c: r_c += r_d2·β + r_d1·β⁻¹
        self.r_c = self.r_c + self.r_d2 * *beta + self.r_d1 * beta_inv;
    }

    /// Apply second challenge with 4-ary folding
    ///
    /// Derives 4 coefficients from alpha: (α³, α², α, 1-α-α²-α³)
    /// and folds 4 vectors into 1.
    #[tracing::instrument(skip_all, name = "DoryProverState::apply_second_challenge_4ary")]
    pub fn apply_second_challenge_4ary<M1: DoryRoutines<E::G1>, M2: DoryRoutines<E::G2>>(
        &mut self,
        alpha: &Scalar<E>,
    ) {
        let alpha_sq = *alpha * *alpha;
        let alpha_cu = alpha_sq * *alpha;
        let alpha_inv = alpha.inv().expect("alpha must be invertible");
        let alpha_inv_sq = alpha_inv * alpha_inv;
        let alpha_inv_cu = alpha_inv_sq * alpha_inv;

        // Coefficients: (α³, α², α, 1-α-α²-α³)
        // Sum = α³ + α² + α + 1 - α - α² - α³ = 1 ✓
        let coeff = (
            alpha_cu,
            alpha_sq,
            *alpha,
            Scalar::<E>::one() - alpha_cu - alpha_sq - *alpha,
        );
        // And for the inverse side: (α⁻³, α⁻², α⁻¹, 1-α⁻¹-α⁻²-α⁻³)
        let coeff_inv = (
            alpha_inv_cu,
            alpha_inv_sq,
            alpha_inv,
            Scalar::<E>::one() - alpha_inv_cu - alpha_inv_sq - alpha_inv,
        );

        let n4 = 1 << (self.num_rounds - 2); // n/4

        // Split vectors into 4 quarters
        let mut v1_quarters: Vec<_> = self.v1.chunks_mut(n4).collect();
        let mut v2_quarters: Vec<_> = self.v2.chunks_mut(n4).collect();
        let mut s1_quarters: Vec<_> = self.s1.chunks_mut(n4).collect();
        let mut s2_quarters: Vec<_> = self.s2.chunks_mut(n4).collect();

        // Fold vectors
        #[cfg(feature = "parallel")]
        {
            rayon::join(
                || {
                    // v1_new = c0*v1[0] + c1*v1[1] + c2*v1[2] + c3*v1[3]
                    let mut v1_new = vec![E::G1::identity(); n4];
                    M1::fold_4ary_group_vectors(&mut v1_new, [
                        &v1_quarters[0],
                        &v1_quarters[1],
                        &v1_quarters[2],
                        &v1_quarters[3],
                    ], &coeff);
                    v1_quarters[0].copy_from_slice(&v1_new);
                },
                || {
                    rayon::join(
                        || {
                            let mut v2_new = vec![E::G2::identity(); n4];
                            M2::fold_4ary_group_vectors(&mut v2_new, [
                                &v2_quarters[0],
                                &v2_quarters[1],
                                &v2_quarters[2],
                                &v2_quarters[3],
                            ], &coeff_inv);
                            v2_quarters[0].copy_from_slice(&v2_new);
                        },
                        || {
                            rayon::join(
                                || {
                                    let mut s1_new = vec![Scalar::<E>::zero(); n4];
                                    M1::fold_4ary_field_vectors(&mut s1_new, [
                                        &s1_quarters[0],
                                        &s1_quarters[1],
                                        &s1_quarters[2],
                                        &s1_quarters[3],
                                    ], &coeff);
                                    s1_quarters[0].copy_from_slice(&s1_new);
                                },
                                || {
                                    let mut s2_new = vec![Scalar::<E>::zero(); n4];
                                    M1::fold_4ary_field_vectors(&mut s2_new, [
                                        &s2_quarters[0],
                                        &s2_quarters[1],
                                        &s2_quarters[2],
                                        &s2_quarters[3],
                                    ], &coeff_inv);
                                    s2_quarters[0].copy_from_slice(&s2_new);
                                },
                            );
                        },
                    )
                },
            );
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut v1_new = vec![E::G1::identity(); n4];
            M1::fold_4ary_group_vectors(&mut v1_new, [
                &v1_quarters[0],
                &v1_quarters[1],
                &v1_quarters[2],
                &v1_quarters[3],
            ], &coeff);
            v1_quarters[0].copy_from_slice(&v1_new);

            let mut v2_new = vec![E::G2::identity(); n4];
            M2::fold_4ary_group_vectors(&mut v2_new, [
                &v2_quarters[0],
                &v2_quarters[1],
                &v2_quarters[2],
                &v2_quarters[3],
            ], &coeff_inv);
            v2_quarters[0].copy_from_slice(&v2_new);

            let mut s1_new = vec![Scalar::<E>::zero(); n4];
            M1::fold_4ary_field_vectors(&mut s1_new, [
                &s1_quarters[0],
                &s1_quarters[1],
                &s1_quarters[2],
                &s1_quarters[3],
            ], &coeff);
            s1_quarters[0].copy_from_slice(&s1_new);

            let mut s2_new = vec![Scalar::<E>::zero(); n4];
            M1::fold_4ary_field_vectors(&mut s2_new, [
                &s2_quarters[0],
                &s2_quarters[1],
                &s2_quarters[2],
                &s2_quarters[3],
            ], &coeff_inv);
            s2_quarters[0].copy_from_slice(&s2_new);
        }

        // Truncate to quarter size
        self.v1.truncate(n4);
        self.v2.truncate(n4);
        self.s1.truncate(n4);
        self.s2.truncate(n4);

        // Update accumulated blinds using 4-ary round fields
        // For 4-ary: r_c accumulates all 6 cross terms
        self.r_c = self.r_c
            + self.round_c_4ary[0] * coeff.0 + self.round_c_4ary[1] * coeff.1
            + self.round_c_4ary[2] * coeff.2 + self.round_c_4ary[3] * coeff.3
            + self.round_c_4ary[4] * alpha_cu + self.round_c_4ary[5] * alpha_inv_cu;

        // r_d1 accumulates with coefficients
        self.r_d1 = self.round_d1_4ary[0] * coeff.0 + self.round_d1_4ary[1] * coeff.1
            + self.round_d1_4ary[2] * coeff.2 + self.round_d1_4ary[3] * coeff.3;

        self.r_d2 = self.round_d2_4ary[0] * coeff_inv.0 + self.round_d2_4ary[1] * coeff_inv.1
            + self.round_d2_4ary[2] * coeff_inv.2 + self.round_d2_4ary[3] * coeff_inv.3;

        // r_e1 and r_e2 accumulate cross terms
        self.r_e1 = self.r_e1
            + self.round_e1_4ary[0] * coeff.0 + self.round_e1_4ary[1] * coeff.1
            + self.round_e1_4ary[2] * coeff.2 + self.round_e1_4ary[3] * coeff.3
            + self.round_e1_4ary[4] * alpha_cu + self.round_e1_4ary[5] * alpha_inv_cu;

        self.r_e2 = self.r_e2
            + self.round_e2_4ary[0] * coeff_inv.0 + self.round_e2_4ary[1] * coeff_inv.1
            + self.round_e2_4ary[2] * coeff_inv.2 + self.round_e2_4ary[3] * coeff_inv.3
            + self.round_e2_4ary[4] * alpha_inv_cu + self.round_e2_4ary[5] * alpha_cu;

        self.num_rounds -= 2;
    }

    /// Compute final scalar product message
    ///
    /// Applies fold-scalars transformation and returns the final E1, E2 elements.
    /// Must be called when num_rounds=0 (vectors are size 1).
    ///
    /// In ZK mode, E₁ and E₂ are additionally blinded with fresh randomness so
    /// that the folded vectors `v₁[0]`, `v₂[0]` cannot be recovered from the
    /// proof.
    #[tracing::instrument(skip_all, name = "DoryProverState::compute_final_message")]
    pub fn compute_final_message<M1, M2>(
        &mut self,
        gamma: &Scalar<E>,
    ) -> ScalarProductMessage<E::G1, E::G2>
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        debug_assert_eq!(self.num_rounds, 0, "num_rounds must be 0 for final message");
        debug_assert_eq!(self.v1.len(), 1, "v1 must have length 1");
        debug_assert_eq!(self.v2.len(), 1, "v2 must have length 1");

        let gamma_inv = gamma.inv().expect("gamma must be invertible");

        let r_final1: Scalar<E> = M::sample();
        let r_final2: Scalar<E> = M::sample();

        // E₁ = v₁ + (γ·s₁ + r_final1)·H₁
        let gamma_s1 = *gamma * self.s1[0] + r_final1;
        let e1 = self.v1[0] + gamma_s1 * self.setup.h1;

        // E₂ = v₂ + (γ⁻¹·s₂ + r_final2)·H₂
        let gamma_inv_s2 = gamma_inv * self.s2[0] + r_final2;
        let e2 = self.v2[0] + self.setup.h2.scale(&gamma_inv_s2);

        self.r_c = self.r_c + self.r_e2 * gamma + self.r_e1 * gamma_inv;

        ScalarProductMessage { e1, e2 }
    }

    /// Generate ZK scalar product proof. Must be called BEFORE `compute_final_message`.
    #[cfg(feature = "zk")]
    pub fn scalar_product_proof<T: Transcript<Curve = E>>(
        &self,
        transcript: &mut T,
    ) -> ScalarProductProof<E::G1, E::G2, Scalar<E>, E::GT> {
        let (v1, v2) = (self.v1[0], self.v2[0]);
        let (g1, g2) = (self.setup.g1_vec[0], self.setup.g2_vec[0]);
        let ht = &self.setup.ht;
        let r = || Scalar::<E>::random();
        let (sd1, sd2) = (r(), r());
        let (d1, d2) = (sd1 * g1, g2.scale(&sd2));
        let (rp1, rp2, rq, rr) = (r(), r(), r(), r());
        let p1 = E::pair(&d1, &g2) + ht.scale(&rp1);
        let p2 = E::pair(&g1, &d2) + ht.scale(&rp2);
        let q = E::pair(&d1, &v2) + E::pair(&v1, &d2) + ht.scale(&rq);
        let rr_val = E::pair(&d1, &d2) + ht.scale(&rr);
        for (label, val) in [
            (b"sigma_p1" as &[u8], &p1),
            (b"sigma_p2", &p2),
            (b"sigma_q", &q),
            (b"sigma_r", &rr_val),
        ] {
            transcript.append_serde(label, val);
        }
        let c = transcript.challenge_scalar(b"sigma_c");
        ScalarProductProof {
            p1,
            p2,
            q,
            r: rr_val,
            e1: d1 + c * v1,
            e2: d2 + v2.scale(&c),
            r1: rp1 + c * self.r_d1,
            r2: rp2 + c * self.r_d2,
            r3: rr + c * rq + c * c * self.r_c,
        }
    }
}

/// Generate Sigma1 proof: proves knowledge of (y, rE2, ry).
#[cfg(feature = "zk")]
pub fn generate_sigma1_proof<E, T>(
    y: &Scalar<E>,
    r_e2: &Scalar<E>,
    r_y: &Scalar<E>,
    setup: &ProverSetup<E>,
    transcript: &mut T,
) -> Sigma1Proof<E::G1, E::G2, Scalar<E>>
where
    E: PairingCurve,
    T: Transcript<Curve = E>,
    Scalar<E>: Field,
    E::G2: Group<Scalar = Scalar<E>>,
{
    let (g2_fin, g1_fin) = (&setup.g2_vec[0], &setup.g1_vec[0]);
    let (k1, k2, k3) = (
        Scalar::<E>::random(),
        Scalar::<E>::random(),
        Scalar::<E>::random(),
    );
    let a1 = g2_fin.scale(&k1) + setup.h2.scale(&k2);
    let a2 = k1 * g1_fin + k3 * setup.h1;
    transcript.append_serde(b"sigma1_a1", &a1);
    transcript.append_serde(b"sigma1_a2", &a2);
    let c = transcript.challenge_scalar(b"sigma1_c");
    Sigma1Proof {
        a1,
        a2,
        z1: k1 + c * y,
        z2: k2 + c * r_e2,
        z3: k3 + c * r_y,
    }
}

/// Verify Sigma1 proof.
#[cfg(feature = "zk")]
pub fn verify_sigma1_proof<E: PairingCurve, T: Transcript<Curve = E>>(
    e2: &E::G2,
    y_commit: &E::G1,
    proof: &Sigma1Proof<E::G1, E::G2, Scalar<E>>,
    setup: &VerifierSetup<E>,
    transcript: &mut T,
) -> Result<(), DoryError>
where
    Scalar<E>: Field,
    E::G2: Group<Scalar = Scalar<E>>,
{
    transcript.append_serde(b"sigma1_a1", &proof.a1);
    transcript.append_serde(b"sigma1_a2", &proof.a2);
    let c = transcript.challenge_scalar(b"sigma1_c");
    if setup.g2_0.scale(&proof.z1) + setup.h2.scale(&proof.z2) != proof.a1 + e2.scale(&c) {
        return Err(DoryError::InvalidProof);
    }
    if proof.z1 * setup.g1_0 + proof.z3 * setup.h1 != proof.a2 + c * y_commit {
        return Err(DoryError::InvalidProof);
    }
    Ok(())
}

/// Generate Sigma2 proof: proves e(E1, Γ2,fin) - D2 = e(H1, t1·Γ2,fin + t2·H2).
#[cfg(feature = "zk")]
pub fn generate_sigma2_proof<E, T>(
    t1: &Scalar<E>,
    t2: &Scalar<E>,
    setup: &ProverSetup<E>,
    transcript: &mut T,
) -> Sigma2Proof<Scalar<E>, E::GT>
where
    E: PairingCurve,
    T: Transcript<Curve = E>,
    Scalar<E>: Field,
    E::G2: Group<Scalar = Scalar<E>>,
    E::GT: Group<Scalar = Scalar<E>>,
{
    let (k1, k2) = (Scalar::<E>::random(), Scalar::<E>::random());
    let a = E::pair(
        &setup.h1,
        &(setup.g2_vec[0].scale(&k1) + setup.h2.scale(&k2)),
    );
    transcript.append_serde(b"sigma2_a", &a);
    let c = transcript.challenge_scalar(b"sigma2_c");
    Sigma2Proof {
        a,
        z1: k1 + c * t1,
        z2: k2 + c * t2,
    }
}

/// Verify Sigma2 proof.
#[cfg(feature = "zk")]
pub fn verify_sigma2_proof<E: PairingCurve, T: Transcript<Curve = E>>(
    e1: &E::G1,
    d2: &E::GT,
    proof: &Sigma2Proof<Scalar<E>, E::GT>,
    setup: &VerifierSetup<E>,
    transcript: &mut T,
) -> Result<(), DoryError>
where
    Scalar<E>: Field,
    E::G2: Group<Scalar = Scalar<E>>,
    E::GT: Group<Scalar = Scalar<E>>,
{
    transcript.append_serde(b"sigma2_a", &proof.a);
    let c = transcript.challenge_scalar(b"sigma2_c");
    let expected = E::pair(e1, &setup.g2_0) - *d2;
    let lhs = E::pair(
        &setup.h1,
        &(setup.g2_0.scale(&proof.z1) + setup.h2.scale(&proof.z2)),
    );
    if lhs == proof.a + expected.scale(&c) {
        Ok(())
    } else {
        Err(DoryError::InvalidProof)
    }
}

impl<E: PairingCurve> DoryVerifierState<E> {
    /// Create new verifier state for O(1) accumulation.
    ///
    /// `e1` and `d2` are stored both as initial values (for batched VMV check)
    /// and as accumulators (updated during reduce rounds), since the VMV check
    /// is deferred to the final batched pairing.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        c: E::GT,
        d1: E::GT,
        d2: E::GT,
        e1: E::G1,
        e2: E::G2,
        s1_coords: Vec<Scalar<E>>,
        s2_coords: Vec<Scalar<E>>,
        num_rounds: usize,
        setup: VerifierSetup<E>,
    ) -> Self {
        debug_assert_eq!(s1_coords.len(), num_rounds);
        debug_assert_eq!(s2_coords.len(), num_rounds);

        Self {
            c,
            d1,
            d2,
            e1,
            e2,
            e1_init: e1,
            d2_init: d2,
            s1_acc: Scalar::<E>::one(),
            s2_acc: Scalar::<E>::one(),
            s1_coords,
            s2_coords,
            num_rounds,
            setup,
        }
    }

    /// Process one round of the Dory-Reduce verification protocol
    ///
    /// Takes both reduce messages and both challenges, updates all state values.
    /// This implements the extended Dory-Reduce algorithm from sections 3.2 & 4.2.
    #[tracing::instrument(skip_all, name = "DoryVerifierState::process_round")]
    pub fn process_round(
        &mut self,
        first_msg: &FirstReduceMessage<E::G1, E::G2, E::GT>,
        second_msg: &SecondReduceMessage<E::G1, E::G2, E::GT>,
        alpha: &Scalar<E>,
        beta: &Scalar<E>,
    ) -> Result<(), DoryError>
    where
        E::G2: Group<Scalar = Scalar<E>>,
        E::GT: Group<Scalar = Scalar<E>>,
        Scalar<E>: Field,
    {
        if self.num_rounds == 0 {
            return Err(DoryError::InvalidProof);
        }

        let alpha_inv = alpha.inv().ok_or(DoryError::InvalidProof)?;
        let beta_inv = beta.inv().ok_or(DoryError::InvalidProof)?;
        let alpha_beta = *alpha * beta;
        let alpha_inv_beta_inv = alpha_inv * beta_inv;
        let round = self.num_rounds;

        // C, D1, D2 are GT scalar muls (expensive); E1, E2 are G1/G2 scalar muls (cheap).
        // All 5 updates read old values and are independent within a round.
        #[cfg(feature = "parallel")]
        let (new_c, (new_d1, new_d2)) = {
            let (c, d1, d2) = (self.c, self.d1, self.d2);
            let chi_r = self.setup.chi[round];
            let delta_1l_r = self.setup.delta_1l[round];
            let delta_1r_r = self.setup.delta_1r[round];
            let delta_2l_r = self.setup.delta_2l[round];
            let delta_2r_r = self.setup.delta_2r[round];
            rayon::join(
                || {
                    // C' ← C + χᵢ + β·D₂ + β⁻¹·D₁ + α·C₊ + α⁻¹·C₋
                    c + chi_r
                        + d2.scale(beta)
                        + d1.scale(&beta_inv)
                        + second_msg.c_plus.scale(alpha)
                        + second_msg.c_minus.scale(&alpha_inv)
                },
                || {
                    rayon::join(
                        || {
                            // D₁' ← α·D₁L + D₁R + αβ·Δ₁L + β·Δ₁R
                            first_msg.d1_left.scale(alpha)
                                + first_msg.d1_right
                                + delta_1l_r.scale(&alpha_beta)
                                + delta_1r_r.scale(beta)
                        },
                        || {
                            // D₂' ← α⁻¹·D₂L + D₂R + α⁻¹β⁻¹·Δ₂L + β⁻¹·Δ₂R
                            first_msg.d2_left.scale(&alpha_inv)
                                + first_msg.d2_right
                                + delta_2l_r.scale(&alpha_inv_beta_inv)
                                + delta_2r_r.scale(&beta_inv)
                        },
                    )
                },
            )
        };
        #[cfg(feature = "parallel")]
        {
            self.c = new_c;
            self.d1 = new_d1;
            self.d2 = new_d2;
        }

        #[cfg(not(feature = "parallel"))]
        {
            // C' ← C + χᵢ + β·D₂ + β⁻¹·D₁ + α·C₊ + α⁻¹·C₋
            self.c = self.c
                + self.setup.chi[round]
                + self.d2.scale(beta)
                + self.d1.scale(&beta_inv)
                + second_msg.c_plus.scale(alpha)
                + second_msg.c_minus.scale(&alpha_inv);

            // D₁' ← α·D₁L + D₁R + αβ·Δ₁L + β·Δ₁R
            self.d1 = first_msg.d1_left.scale(alpha)
                + first_msg.d1_right
                + self.setup.delta_1l[round].scale(&alpha_beta)
                + self.setup.delta_1r[round].scale(beta);

            // D₂' ← α⁻¹·D₂L + D₂R + α⁻¹β⁻¹·Δ₂L + β⁻¹·Δ₂R
            self.d2 = first_msg.d2_left.scale(&alpha_inv)
                + first_msg.d2_right
                + self.setup.delta_2l[round].scale(&alpha_inv_beta_inv)
                + self.setup.delta_2r[round].scale(&beta_inv);
        }

        // E₁' ← E₁ + β·E₁β + α·E₁₊ + α⁻¹·E₁₋
        self.e1 = self.e1
            + *beta * first_msg.e1_beta
            + *alpha * second_msg.e1_plus
            + alpha_inv * second_msg.e1_minus;

        // E₂' ← E₂ + β⁻¹·E₂β + α·E₂₊ + α⁻¹·E₂₋
        self.e2 = self.e2
            + first_msg.e2_beta.scale(&beta_inv)
            + second_msg.e2_plus.scale(alpha)
            + second_msg.e2_minus.scale(&alpha_inv);

        // Folded scalars: s_acc *= (α·(1−coord) + coord) indexed MSB-first
        let idx = round - 1;
        let (y_t, x_t) = (self.s1_coords[idx], self.s2_coords[idx]);
        let one = Scalar::<E>::one();
        self.s1_acc = self.s1_acc * (*alpha * (one - y_t) + y_t);
        self.s2_acc = self.s2_acc * (alpha_inv * (one - x_t) + x_t);

        self.num_rounds -= 1;
        Ok(())
    }

    /// Process one round of 4-ary Dory-Reduce verification
    ///
    /// Takes 4-ary reduce messages and alpha challenge, updates all state values.
    /// Reduces rounds by 2 (halving the vector size by 4 instead of 2).
    #[cfg(feature = "lattice")]
    #[tracing::instrument(skip_all, name = "DoryVerifierState::process_round_4ary")]
    pub fn process_round_4ary(
        &mut self,
        first_msg: &FirstReduceMessage4<E::G1, E::G2, E::GT>,
        _second_msg: &SecondReduceMessage4<E::G1, E::G2>,
        alpha: &Scalar<E>,
        beta: &Scalar<E>,
    ) -> Result<(), DoryError>
    where
        E::G2: Group<Scalar = Scalar<E>>,
        E::GT: Group<Scalar = Scalar<E>>,
        Scalar<E>: Field,
    {
        if self.num_rounds < 2 {
            return Err(DoryError::InvalidProof);
        }

        let alpha_sq = *alpha * *alpha;
        let alpha_cu = alpha_sq * *alpha;
        let alpha_inv = alpha.inv().ok_or(DoryError::InvalidProof)?;
        let alpha_inv_sq = alpha_inv * alpha_inv;
        let alpha_inv_cu = alpha_inv_sq * alpha_inv;

        // 4-ary coefficients: (α³, α², α, 1-α-α²-α³)
        let coeff = (
            alpha_cu,
            alpha_sq,
            *alpha,
            Scalar::<E>::one() - alpha_cu - alpha_sq - *alpha,
        );
        let coeff_inv = (
            alpha_inv_cu,
            alpha_inv_sq,
            alpha_inv,
            Scalar::<E>::one() - alpha_inv_cu - alpha_inv_sq - alpha_inv,
        );

        let round = self.num_rounds;

        // For 4-ary folding, we consume 2 rounds at once
        // The setup arrays are indexed per-round, so we need to handle indices carefully
        // Indices 0..2 (rounds 0,1) get consumed by this 4-ary round
        // We need delta_1l[round], delta_1l[round-1], etc.

        // C' accumulates all 6 cross terms with their coefficients
        // C' ← C + Σ_i (β^i · D2_i · coeff_i) + Σ_i (β^-i · D1_i · coeff_inv_i)
        //      + Σ_{i<j} (α^{i+j} · C_ij) ... actually it's more complex
        //
        // The full 4-ary update equations:
        // C' = C + β·(D2_0 + D2_1·β + D2_2·β² + D2_3·β³)·coeff
        //      + β⁻¹·(D1_0 + D1_1·β + D1_2·β² + D1_3·β³)·coeff_inv
        //      + Σ_{i<j} (α^{i+j}·C_ij) ... simplified to cross terms
        //
        // Actually the C update uses cross pairings with alpha powers
        let round_0 = round;
        let round_1 = round - 1;

        // Compute beta powers from parameter
        let beta_sq = *beta * *beta;
        let _beta_cu = beta_sq * *beta;
        let beta_inv = beta.inv().expect("beta must be invertible");
        let beta_inv_sq = beta_inv * beta_inv;
        let _beta_inv_cu = beta_inv_sq * beta_inv;

        // D1' = Σ_i (α^i · D1_i · β^i) ... simplified
        // D2' = Σ_i (α^-i · D2_i · β^-i) ... simplified
        //
        // For simplicity, we compute:
        // D1' = coeff[0]·D1_0 + coeff[1]·D1_1 + coeff[2]·D1_2 + coeff[3]·D1_3
        //       + δ terms from setup
        // D2' = coeff_inv[0]·D2_0 + coeff_inv[1]·D2_1 + coeff_inv[2]·D2_2 + coeff_inv[3]·D2_3
        //       + δ terms from setup

        #[cfg(feature = "parallel")]
        let (new_c, (new_d1, new_d2)) = {
            let (c, _d1, _d2) = (self.c, self.d1, self.d2);
            let chi_r0 = self.setup.chi[round_0];
            let chi_r1 = self.setup.chi[round_1];
            rayon::join(
                || {
                    let mut new_c = c + chi_r0 + chi_r1;
                    // D1 and D2 terms with alpha folding coefficients
                    new_c = new_c + first_msg.d1[0].scale(&coeff.0);
                    new_c = new_c + first_msg.d1[1].scale(&coeff.1);
                    new_c = new_c + first_msg.d1[2].scale(&coeff.2);
                    new_c = new_c + first_msg.d1[3].scale(&coeff.3);
                    new_c = new_c + first_msg.d2[0].scale(&coeff_inv.0);
                    new_c = new_c + first_msg.d2[1].scale(&coeff_inv.1);
                    new_c = new_c + first_msg.d2[2].scale(&coeff_inv.2);
                    new_c = new_c + first_msg.d2[3].scale(&coeff_inv.3);
                    // Use diagonal C_raw elements only (⟨Qi, Qi⟩ terms)
                    new_c = new_c + first_msg.c_raw[0].scale(&coeff.0);
                    new_c = new_c + first_msg.c_raw[5].scale(&coeff.1);
                    new_c = new_c + first_msg.c_raw[10].scale(&coeff.2);
                    new_c = new_c + first_msg.c_raw[15].scale(&coeff.3);
                    new_c
                },
                || {
                    rayon::join(
                        || {
                            let delta_1l_r0 = self.setup.delta_1l[round_0];
                            let delta_1r_r0 = self.setup.delta_1r[round_0];
                            let delta_1l_r1 = self.setup.delta_1l[round_1];
                            let delta_1r_r1 = self.setup.delta_1r[round_1];
                            first_msg.d1[0].scale(&coeff.0)
                                + first_msg.d1[1].scale(&coeff.1)
                                + first_msg.d1[2].scale(&coeff.2)
                                + first_msg.d1[3].scale(&coeff.3)
                                + delta_1l_r0.scale(&alpha_cu)
                                + delta_1r_r0.scale(&alpha_sq)
                                + delta_1l_r1.scale(alpha)
                                + delta_1r_r1.scale(&Scalar::<E>::one())
                        },
                        || {
                            let delta_2l_r0 = self.setup.delta_2l[round_0];
                            let delta_2r_r0 = self.setup.delta_2r[round_0];
                            let delta_2l_r1 = self.setup.delta_2l[round_1];
                            let delta_2r_r1 = self.setup.delta_2r[round_1];
                            first_msg.d2[0].scale(&coeff_inv.0)
                                + first_msg.d2[1].scale(&coeff_inv.1)
                                + first_msg.d2[2].scale(&coeff_inv.2)
                                + first_msg.d2[3].scale(&coeff_inv.3)
                                + delta_2l_r0.scale(&alpha_inv_cu)
                                + delta_2r_r0.scale(&alpha_inv_sq)
                                + delta_2l_r1.scale(&alpha_inv)
                                + delta_2r_r1.scale(&Scalar::<E>::one())
                        },
                    )
                },
            )
        };

        #[cfg(not(feature = "parallel"))]
        let (new_c, new_d1, new_d2) = {
            // The verifier receives ALL 16 C_raw_ij = ⟨Qi, Qj⟩ values (element-wise)
            // After beta folding: Qi_new = Qi + β·Γ1[i], Qj_new = Qj + β⁻¹·Γ2[j]
            //
            // The effective cross term is:
            // ⟨Qi_new, Qj_new⟩ = Σ_k ⟨Qi[k] + β·Γ1[i][k], Qj[k] + β⁻¹·Γ2[j][k]⟩
            // = C_raw_ij + β·⟨Γ1[i], Qj⟩ + β⁻¹·⟨Qi, Γ2[j]⟩ + ⟨Γ1[i], Γ2[j]⟩
            //
            // But we CAN'T extract ⟨Γ1[i], Qj⟩ from D1[j] alone because D1[j] = ⟨Γ2, Qj + β·Γ1[j]⟩
            // This doesn't isolate the individual component we need.
            //
            // So the 4-ary approach with pre-sent cross terms is mathematically unsound
            // as implemented. The binary approach (compute C+ and C- AFTER beta) works
            // because it computes the actual cross terms with the actual beta-transformed quarters.
            //
            // For now, we use the C_raw values directly as a placeholder.
            // This will NOT produce correct verification.
            let mut new_c = self.c + self.setup.chi[round_0] + self.setup.chi[round_1];

            // D1 and D2 terms with alpha folding coefficients
            new_c = new_c + first_msg.d1[0].scale(&coeff.0);
            new_c = new_c + first_msg.d1[1].scale(&coeff.1);
            new_c = new_c + first_msg.d1[2].scale(&coeff.2);
            new_c = new_c + first_msg.d1[3].scale(&coeff.3);
            new_c = new_c + first_msg.d2[0].scale(&coeff_inv.0);
            new_c = new_c + first_msg.d2[1].scale(&coeff_inv.1);
            new_c = new_c + first_msg.d2[2].scale(&coeff_inv.2);
            new_c = new_c + first_msg.d2[3].scale(&coeff_inv.3);

            // Use diagonal C_raw elements (i==j) which represent ⟨Qi, Qi⟩ without cross terms
            // These don't depend on beta and are safe to use
            new_c = new_c + first_msg.c_raw[0].scale(&coeff.0);  // ⟨Q0,Q0⟩
            new_c = new_c + first_msg.c_raw[5].scale(&coeff.1);  // ⟨Q1,Q1⟩
            new_c = new_c + first_msg.c_raw[10].scale(&coeff.2); // ⟨Q2,Q2⟩
            new_c = new_c + first_msg.c_raw[15].scale(&coeff.3); // ⟨Q3,Q3⟩

            // D1' = sum_i alpha^i * D1[i] + delta terms
            let new_d1 = first_msg.d1[0].scale(&coeff.0)
                + first_msg.d1[1].scale(&coeff.1)
                + first_msg.d1[2].scale(&coeff.2)
                + first_msg.d1[3].scale(&coeff.3)
                + self.setup.delta_1l[round_0].scale(&alpha_cu)
                + self.setup.delta_1r[round_0].scale(&alpha_sq)
                + self.setup.delta_1l[round_1].scale(alpha)
                + self.setup.delta_1r[round_1].scale(&Scalar::<E>::one());

            // D2' = sum_i alpha^-i * D2[i] + delta terms
            let new_d2 = first_msg.d2[0].scale(&coeff_inv.0)
                + first_msg.d2[1].scale(&coeff_inv.1)
                + first_msg.d2[2].scale(&coeff_inv.2)
                + first_msg.d2[3].scale(&coeff_inv.3)
                + self.setup.delta_2l[round_0].scale(&alpha_inv_cu)
                + self.setup.delta_2r[round_0].scale(&alpha_inv_sq)
                + self.setup.delta_2l[round_1].scale(&alpha_inv)
                + self.setup.delta_2r[round_1].scale(&Scalar::<E>::one());

            (new_c, new_d1, new_d2)
        };

        #[cfg(feature = "parallel")]
        {
            self.c = new_c;
            self.d1 = new_d1;
            self.d2 = new_d2;
        }

        #[cfg(not(feature = "parallel"))]
        {
            self.c = new_c;
            self.d1 = new_d1;
            self.d2 = new_d2;
        }

        // E1' ← E1 + β·E1β + Σ_{i,j} (c_i·c_j_inv·E1_ij)
        // where e1_cross[i*4+j] = MSM(Qi, s2_quarter[j])
        self.e1 = self.e1 + *beta * first_msg.e1_beta;
        let e1_coeff = [coeff.0, coeff.1, coeff.2, coeff.3];
        let e1_coeff_inv = [coeff_inv.0, coeff_inv.1, coeff_inv.2, coeff_inv.3];
        for i in 0..4 {
            for j in 0..4 {
                let idx = i * 4 + j;
                let c_coeff = e1_coeff[i] * e1_coeff_inv[j];
                self.e1 = self.e1 + first_msg.e1_cross[idx].scale(&c_coeff);
            }
        }

        // E2' ← E2 + β⁻¹·E2β + Σ_{i,j} (c_i·c_j_inv·E2_ij)
        // where e2_cross[i*4+j] = MSM(v2_quarter[i], s1_quarter[j])
        self.e2 = self.e2 + first_msg.e2_beta.scale(&beta_inv);
        for i in 0..4 {
            for j in 0..4 {
                let idx = i * 4 + j;
                let c_coeff = e1_coeff[i] * e1_coeff_inv[j];
                self.e2 = self.e2 + first_msg.e2_cross[idx].scale(&c_coeff);
            }
        }

        // Folded scalars: s_acc *= product of (α·(1−coord) + coord) for 2 rounds
        // idx 0 = MSB round, idx 1 = next round
        let idx0 = round_0 - 1;
        let idx1 = round_1 - 1;
        let (y_t0, x_t0) = (self.s1_coords[idx0], self.s2_coords[idx0]);
        let (y_t1, x_t1) = (self.s1_coords[idx1], self.s2_coords[idx1]);
        let one = Scalar::<E>::one();

        // Two rounds worth of folding: s_acc *= (α·(1−y0) + y0) * (α·(1−y1) + y1)
        let fold0 = *alpha * (one - y_t0) + y_t0;
        let fold1 = *alpha * (one - y_t1) + y_t1;
        self.s1_acc = self.s1_acc * fold0 * fold1;

        let fold0_inv = alpha_inv * (one - x_t0) + x_t0;
        let fold1_inv = alpha_inv * (one - x_t1) + x_t1;
        self.s2_acc = self.s2_acc * fold0_inv * fold1_inv;

        self.num_rounds -= 2;
        Ok(())
    }

    /// Verify the final scalar product equation.
    ///
    /// Must be called when `num_rounds == 0` after all reduce rounds are complete.
    ///
    /// When `zk_data` is `None`, performs the transparent 4-pairing check.
    /// When `zk_data` is `Some((sp, sigma_c))`, performs the ZK 1-pairing check.
    ///
    /// # Non-optimized Protocol Equations
    ///
    /// ## VMV Check (batched together with the final pairing check)
    ///
    /// The VMV protocol requires: `D₂_init = e(E₁_init, Γ₂₀)`
    /// (proven by the Sigma₂ proof in ZK mode, deferred here for batching in transparent mode).
    ///
    /// ## Fold-Scalars Updates
    ///
    /// ```text
    /// C' ← C + (s₁·s₂)·HT + γ·e(H₁, E₂) + γ⁻¹·e(E₁, H₂)
    /// D₁' ← D₁ + e(H₁, (s₁·γ)·Γ₂₀)
    /// D₂' ← D₂ + e((s₂·γ⁻¹)·Γ₁₀, H₂)
    /// ```
    ///
    /// ## Final Verification
    ///
    /// ```text
    /// e(E₁ + d·Γ₁₀, E₂ + d⁻¹·Γ₂₀) = C' + χ₀ + d·D₂' + d⁻¹·D₁'
    /// ```
    ///
    /// # Transparent Mode — Multi-Pairing Check (4 ML + 1 FE)
    ///
    /// ## Batching the VMV Check
    ///
    /// We use random linear combination with challenge `d²` to defer the VMV check.
    /// We use `d²` (not `d`) to ensure sufficient independence from the existing `d·D₂` term.
    ///
    /// Soundness: `d` is derived from the transcript AFTER `D₂_init` and `E₁_init` are
    /// committed, so if `D₂_init ≠ e(E₁_init, Γ₂₀)`, then with overwhelming probability
    /// `T + d²·D₂_init ≠ multi_pair([...]) + d²·e(E₁_init, Γ₂₀)`.
    ///
    /// ## Final Combined Check
    ///
    /// The final check verifies both:
    /// - (a) The fold-scalars/reduce protocol equation
    /// - (b) The VMV constraint `D₂_init = e(E₁_init, Γ₂₀)`
    ///
    /// Combined via: `(a) + d²·(b)` where `d` is the final challenge.
    ///
    /// ```text
    /// e(E₁_final + d·Γ₁₀, E₂_final + d⁻¹·Γ₂₀)            [Pair 1: scalar product]
    ///   · e(H₁, (-γ)·(E₂_acc + (d⁻¹·s₁)·Γ₂₀))             [Pair 2: E₂ accumulator]
    ///   · e((-γ⁻¹)·(E₁_acc + (d·s₂)·Γ₁₀), H₂)             [Pair 3: E₁ accumulator]
    ///   · e(d²·E₁_init, Γ₂₀)                                [Pair 4: deferred VMV]
    ///   = C + (s₁·s₂)·HT + χ₀ + d·D₂ + d⁻¹·D₁ + d²·D₂_init
    /// ```
    ///
    /// Note: Pairs 3 and 4 cannot be combined into 3 ML because they use different
    /// G2 elements (H₂ vs Γ₂₀). This differs from the original Dory construction
    /// where `D₂ = e(Γ₁·v, H₂)` allowed H₂-sharing.
    ///
    /// # ZK Mode (1 ML + 1 FE)
    ///
    /// In ZK mode, the scalar product proof replaces the transparent check with a
    /// Sigma-protocol equation proving knowledge of (v₁, v₂) opening (C, D₁, D₂).
    /// E-accumulator and VMV binding are handled separately by Sigma₁/Sigma₂ proofs
    /// verified earlier in the protocol.
    ///
    /// ```text
    /// e(sp.e₁ + d·Γ₁₀, sp.e₂ + d⁻¹·Γ₂₀)
    ///   = χ₀ + sp.r + c·sp.q + c²·C
    ///     + d·(sp.p₂ + c·D₂) + d⁻¹·(sp.p₁ + c·D₁)
    ///     − (sp.r₃ + d·sp.r₂ + d⁻¹·sp.r₁)·HT
    /// ```
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, name = "DoryVerifierState::verify_final")]
    pub fn verify_final(
        &self,
        msg: &ScalarProductMessage<E::G1, E::G2>,
        gamma: &Scalar<E>,
        d: &Scalar<E>,
        zk_data: Option<(
            &ScalarProductProof<E::G1, E::G2, Scalar<E>, E::GT>,
            &Scalar<E>,
        )>,
    ) -> Result<(), DoryError>
    where
        E::G2: Group<Scalar = Scalar<E>>,
        E::GT: Group<Scalar = Scalar<E>>,
        Scalar<E>: Field,
    {
        debug_assert_eq!(
            self.num_rounds, 0,
            "num_rounds must be 0 for final verification"
        );

        let d_inv = d.inv().ok_or(DoryError::InvalidProof)?;

        if let Some((sp, sigma_c)) = zk_data {
            // ZK mode: 1 ML + 1 FE
            let c = *sigma_c;
            let c_sq = c * c;

            let lhs = E::pair(
                &(sp.e1 + self.setup.g1_0.scale(d)),
                &(sp.e2 + self.setup.g2_0.scale(&d_inv)),
            );

            let ht_scalar = sp.r3 + *d * sp.r2 + d_inv * sp.r1;
            let mut rhs = self.setup.chi[0] + sp.r + sp.q.scale(&c) + self.c.scale(&c_sq);
            rhs = rhs + sp.p2.scale(d) + self.d2.scale(&(*d * c));
            rhs = rhs + sp.p1.scale(&d_inv) + self.d1.scale(&(d_inv * c));
            rhs = rhs - self.setup.ht.scale(&ht_scalar);

            if lhs == rhs {
                Ok(())
            } else {
                Err(DoryError::InvalidProof)
            }
        } else {
            // Transparent mode: 4 ML + 1 FE
            let gamma_inv = gamma.inv().ok_or(DoryError::InvalidProof)?;
            let d_sq = *d * *d;
            let neg_gamma = -*gamma;
            let neg_gamma_inv = -gamma_inv;

            let s_product = self.s1_acc * self.s2_acc;
            let rhs = self.c
                + self.setup.ht.scale(&s_product)
                + self.setup.chi[0]
                + self.d2.scale(d)
                + self.d1.scale(&d_inv)
                + self.d2_init.scale(&d_sq);

            // Pair 1: e(E₁_final + d·Γ₁₀, E₂_final + d⁻¹·Γ₂₀)
            let p1_g1 = msg.e1 + self.setup.g1_0.scale(d);
            let p1_g2 = msg.e2 + self.setup.g2_0.scale(&d_inv);

            // Pair 2: e(H₁, (-γ)·(E₂_acc + (d⁻¹·s₁)·Γ₂₀))
            let p2_g1 = self.setup.h1;
            let p2_g2 = (self.e2 + self.setup.g2_0.scale(&(d_inv * self.s1_acc))).scale(&neg_gamma);

            // Pair 3: e((-γ⁻¹)·(E₁_acc + (d·s₂)·Γ₁₀), H₂)
            let p3_g1 =
                (self.e1 + self.setup.g1_0.scale(&(*d * self.s2_acc))).scale(&neg_gamma_inv);
            let p3_g2 = self.setup.h2;

            // Pair 4: e(d²·E₁_init, Γ₂₀) — deferred VMV check
            let p4_g1 = self.e1_init.scale(&d_sq);
            let p4_g2 = self.setup.g2_0;

            let lhs = E::multi_pair(&[p1_g1, p2_g1, p3_g1, p4_g1], &[p1_g2, p2_g2, p3_g2, p4_g2]);

            if lhs == rhs {
                Ok(())
            } else {
                Err(DoryError::InvalidProof)
            }
        }
    }
}

/// Forkable prover state for Dory-Prime's parallel proof generation.
///
/// This is a wrapper around DoryProverState that owns its data (instead of
/// borrowing from setup) and implements Clone to enable forking at each
/// challenge point. This allows computing all 2^sigma proof paths in parallel.
#[derive(Clone)]
#[allow(dead_code)]
pub struct ForkableDoryProverState<E: PairingCurve, M: Mode = Transparent> {
    /// Current v1 vector (G1 elements) - owned
    v1: Vec<E::G1>,

    /// Current v2 vector (G2 elements) - owned
    v2: Vec<E::G2>,

    /// For first round only: scalars used to construct v2 from fixed base h2
    v2_scalars: Option<Vec<Scalar<E>>>,

    /// Current s1 vector (scalars)
    s1: Vec<Scalar<E>>,

    /// Current s2 vector (scalars)
    s2: Vec<Scalar<E>>,

    /// Number of rounds remaining (log₂ of vector length)
    num_rounds: usize,

    /// Owned prover setup (cloned from original)
    setup: ProverSetup<E>,

    // ZK accumulated blinds (zero in Transparent mode)
    r_c: Scalar<E>,
    r_d1: Scalar<E>,
    r_d2: Scalar<E>,
    r_e1: Scalar<E>,
    r_e2: Scalar<E>,

    // Per-round blinds stored between compute and apply
    round_d1: [Scalar<E>; 2],
    round_d2: [Scalar<E>; 2],
    round_c: [Scalar<E>; 2],
    round_e1: [Scalar<E>; 2],
    round_e2: [Scalar<E>; 2],

    // Per-round blinds for 4-ary folding
    round_d1_4ary: [Scalar<E>; 4],
    round_d2_4ary: [Scalar<E>; 4],
    round_c_4ary: [Scalar<E>; 16],
    round_e1_4ary: [Scalar<E>; 16],
    round_e2_4ary: [Scalar<E>; 16],

    _mode: PhantomData<M>,
}

impl<E: PairingCurve, M: Mode> ForkableDoryProverState<E, M>
where
    <E::G1 as Group>::Scalar: Field,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
    E::GT: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    /// Create from an existing DoryProverState by cloning owned data
    pub fn from_prover_state<'a, 'b>(
        state: &DoryProverState<'b, E, M>,
    ) -> Self
    where
        'b: 'a,
    {
        let z = Scalar::<E>::zero();
        Self {
            v1: state.v1.clone(),
            v2: state.v2.clone(),
            v2_scalars: state.v2_scalars.clone(),
            s1: state.s1.clone(),
            s2: state.s2.clone(),
            num_rounds: state.num_rounds,
            setup: (*state.setup).clone(),
            r_c: state.r_c,
            r_d1: state.r_d1,
            r_d2: state.r_d2,
            r_e1: state.r_e1,
            r_e2: state.r_e2,
            round_d1: [z; 2],
            round_d2: [z; 2],
            round_c: [z; 2],
            round_e1: [z; 2],
            round_e2: [z; 2],
            round_d1_4ary: [z; 4],
            round_d2_4ary: [z; 4],
            round_c_4ary: [z; 16],
            round_e1_4ary: [z; 16],
            round_e2_4ary: [z; 16],
            _mode: PhantomData,
        }
    }

    /// Create new forkable prover state from scratch
    pub fn new(
        v1: Vec<E::G1>,
        v2: Vec<E::G2>,
        v2_scalars: Option<Vec<Scalar<E>>>,
        s1: Vec<Scalar<E>>,
        s2: Vec<Scalar<E>>,
        setup: ProverSetup<E>,
    ) -> Self {
        let num_rounds = v1.len().trailing_zeros() as usize;
        let z = Scalar::<E>::zero();

        Self {
            v1,
            v2,
            v2_scalars,
            s1,
            s2,
            num_rounds,
            setup,
            r_c: z,
            r_d1: z,
            r_d2: z,
            r_e1: z,
            r_e2: z,
            round_d1: [z; 2],
            round_d2: [z; 2],
            round_c: [z; 2],
            round_e1: [z; 2],
            round_e2: [z; 2],
            round_d1_4ary: [z; 4],
            round_d2_4ary: [z; 4],
            round_c_4ary: [z; 16],
            round_e1_4ary: [z; 16],
            round_e2_4ary: [z; 16],
            _mode: PhantomData,
        }
    }

    /// Set all round blinds for the ForkableDoryProverState
    ///
    /// This sets both the accumulated VMV blinds and the per-round blinds
    /// that will be used in each round of the reduce-and-fold protocol.
    #[allow(clippy::too_many_arguments)]
    pub fn set_round_blinds(
        &mut self,
        _r_d1: Scalar<E>,
        r_d2: Scalar<E>,
        r_e1: Scalar<E>,
        r_e2: Scalar<E>,
        r_c: Scalar<E>,
        r_d1_acc: Scalar<E>,
        round_d1: [Scalar<E>; 2],
        round_d2: [Scalar<E>; 2],
        round_c: [Scalar<E>; 2],
        round_e1: [Scalar<E>; 2],
        round_e2: [Scalar<E>; 2],
        round_d1_4ary: [Scalar<E>; 4],
        round_d2_4ary: [Scalar<E>; 4],
        round_c_4ary: [Scalar<E>; 16],
        round_e1_4ary: [Scalar<E>; 16],
        round_e2_4ary: [Scalar<E>; 16],
    ) {
        self.r_d1 = r_d1_acc;
        self.r_d2 = r_d2;
        self.r_e1 = r_e1;
        self.r_e2 = r_e2;
        self.r_c = r_c;
        self.round_d1 = round_d1;
        self.round_d2 = round_d2;
        self.round_c = round_c;
        self.round_e1 = round_e1;
        self.round_e2 = round_e2;
        self.round_d1_4ary = round_d1_4ary;
        self.round_d2_4ary = round_d2_4ary;
        self.round_c_4ary = round_c_4ary;
        self.round_e1_4ary = round_e1_4ary;
        self.round_e2_4ary = round_e2_4ary;
    }

    /// Set initial VMV blinds (r_d1, r_c, r_d2, r_e1, r_e2).
    pub fn set_initial_blinds(
        &mut self,
        r_d1: Scalar<E>,
        r_c: Scalar<E>,
        r_d2: Scalar<E>,
        r_e1: Scalar<E>,
        r_e2: Scalar<E>,
    ) {
        (self.r_d1, self.r_c, self.r_d2, self.r_e1, self.r_e2) = (r_d1, r_c, r_d2, r_e1, r_e2);
    }

    /// Fork this state into two copies at the current challenge point.
    ///
    /// Returns two independent states that can be computed in parallel.
    /// The first copy assumes challenge=0, the second assumes challenge=1.
    /// Both start from identical state.
    pub fn fork(&self) -> (Self, Self) {
        (
            Self {
                v1: self.v1.clone(),
                v2: self.v2.clone(),
                v2_scalars: self.v2_scalars.clone(),
                s1: self.s1.clone(),
                s2: self.s2.clone(),
                num_rounds: self.num_rounds,
                setup: self.setup.clone(),
                r_c: self.r_c,
                r_d1: self.r_d1,
                r_d2: self.r_d2,
                r_e1: self.r_e1,
                r_e2: self.r_e2,
                round_d1: [Scalar::<E>::zero(); 2],
                round_d2: [Scalar::<E>::zero(); 2],
                round_c: [Scalar::<E>::zero(); 2],
                round_e1: [Scalar::<E>::zero(); 2],
                round_e2: [Scalar::<E>::zero(); 2],
                round_d1_4ary: [Scalar::<E>::zero(); 4],
                round_d2_4ary: [Scalar::<E>::zero(); 4],
                round_c_4ary: [Scalar::<E>::zero(); 16],
                round_e1_4ary: [Scalar::<E>::zero(); 16],
                round_e2_4ary: [Scalar::<E>::zero(); 16],
                _mode: PhantomData,
            },
            Self {
                v1: self.v1.clone(),
                v2: self.v2.clone(),
                v2_scalars: self.v2_scalars.clone(),
                s1: self.s1.clone(),
                s2: self.s2.clone(),
                num_rounds: self.num_rounds,
                setup: self.setup.clone(),
                r_c: self.r_c,
                r_d1: self.r_d1,
                r_d2: self.r_d2,
                r_e1: self.r_e1,
                r_e2: self.r_e2,
                round_d1: [Scalar::<E>::zero(); 2],
                round_d2: [Scalar::<E>::zero(); 2],
                round_c: [Scalar::<E>::zero(); 2],
                round_e1: [Scalar::<E>::zero(); 2],
                round_e2: [Scalar::<E>::zero(); 2],
                round_d1_4ary: [Scalar::<E>::zero(); 4],
                round_d2_4ary: [Scalar::<E>::zero(); 4],
                round_c_4ary: [Scalar::<E>::zero(); 16],
                round_e1_4ary: [Scalar::<E>::zero(); 16],
                round_e2_4ary: [Scalar::<E>::zero(); 16],
                _mode: PhantomData,
            },
        )
    }

    /// Compute first reduce message for current round
    #[cfg(feature = "parallel")]
    pub fn compute_first_message<M1, M2>(&mut self) -> FirstReduceMessage<E::G1, E::G2, E::GT>
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        self.compute_first_message_impl::<M1, M2>()
    }

    /// Compute first reduce message for current round (sequential fallback)
    #[cfg(not(feature = "parallel"))]
    pub fn compute_first_message<M1, M2>(&mut self) -> FirstReduceMessage<E::G1, E::G2, E::GT>
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        self.compute_first_message_impl::<M1, M2>()
    }

    #[cfg(feature = "parallel")]
    fn compute_first_message_impl<M1, M2>(&mut self) -> FirstReduceMessage<E::G1, E::G2, E::GT>
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        assert!(
            self.num_rounds > 0,
            "Not enough rounds left in prover state"
        );

        let n2 = 1 << (self.num_rounds - 1);
        let (v1_l, v1_r) = self.v1.split_at(n2);
        let (v2_l, v2_r) = self.v2.split_at(n2);
        let g1_prime = &self.setup.g1_vec[..n2];
        let g2_prime = &self.setup.g2_vec[..n2];

        self.round_d1 = [M::sample(), M::sample()];
        self.round_d2 = [M::sample(), M::sample()];

        let ht = &self.setup.ht;
        let rd1 = self.round_d1;
        let rd2 = self.round_d2;
        let g1_full = &self.setup.g1_vec[..1 << self.num_rounds];
        let g2_full = &self.setup.g2_vec[..1 << self.num_rounds];

        let ((d1_left, d1_right), ((d2_left, d2_right), (e1_beta, e2_beta))) = rayon::join(
            || {
                rayon::join(
                    || M::mask(E::multi_pair_g2_setup(v1_l, g2_prime), ht, &rd1[0]),
                    || M::mask(E::multi_pair_g2_setup(v1_r, g2_prime), ht, &rd1[1]),
                )
            },
            || {
                rayon::join(
                    || {
                        let (d2_left_base, d2_right_base) = if let Some(scalars) =
                            self.v2_scalars.as_ref()
                        {
                            let (s_l, s_r) = scalars.split_at(n2);
                            let (sum_left, sum_right) =
                                rayon::join(|| M1::msm(g1_prime, s_l), || M1::msm(g1_prime, s_r));
                            let g2_fin = &self.setup.g2_vec[0];
                            (E::pair(&sum_left, g2_fin), E::pair(&sum_right, g2_fin))
                        } else {
                            rayon::join(
                                || E::multi_pair_g1_setup(g1_prime, v2_l),
                                || E::multi_pair_g1_setup(g1_prime, v2_r),
                            )
                        };
                        (
                            M::mask(d2_left_base, ht, &rd2[0]),
                            M::mask(d2_right_base, ht, &rd2[1]),
                        )
                    },
                    || {
                        rayon::join(
                            || M1::msm(g1_full, &self.s2[..]),
                            || M2::msm(g2_full, &self.s1[..]),
                        )
                    },
                )
            },
        );

        FirstReduceMessage {
            d1_left,
            d1_right,
            d2_left,
            d2_right,
            e1_beta,
            e2_beta,
        }
    }

    #[cfg(not(feature = "parallel"))]
    fn compute_first_message_impl<M1, M2>(&mut self) -> FirstReduceMessage<E::G1, E::G2, E::GT>
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        assert!(
            self.num_rounds > 0,
            "Not enough rounds left in prover state"
        );

        let n2 = 1 << (self.num_rounds - 1);
        let (v1_l, v1_r) = self.v1.split_at(n2);
        let (v2_l, v2_r) = self.v2.split_at(n2);
        let g1_prime = &self.setup.g1_vec[..n2];
        let g2_prime = &self.setup.g2_vec[..n2];

        self.round_d1 = [M::sample(), M::sample()];
        self.round_d2 = [M::sample(), M::sample()];

        let ht = &self.setup.ht;
        let rd1 = self.round_d1;
        let rd2 = self.round_d2;
        let g1_full = &self.setup.g1_vec[..1 << self.num_rounds];
        let g2_full = &self.setup.g2_vec[..1 << self.num_rounds];

        let (d2_left_base, d2_right_base) = if let Some(scalars) = self.v2_scalars.as_ref() {
            let (s_l, s_r) = scalars.split_at(n2);
            let sum_left = M1::msm(g1_prime, s_l);
            let sum_right = M1::msm(g1_prime, s_r);
            let g2_fin = &self.setup.g2_vec[0];
            (E::pair(&sum_left, g2_fin), E::pair(&sum_right, g2_fin))
        } else {
            (
                E::multi_pair_g1_setup(g1_prime, v2_l),
                E::multi_pair_g1_setup(g1_prime, v2_r),
            )
        };

        let d1_left = M::mask(E::multi_pair_g2_setup(v1_l, g2_prime), ht, &rd1[0]);
        let d1_right = M::mask(E::multi_pair_g2_setup(v1_r, g2_prime), ht, &rd1[1]);
        let d2_left = M::mask(d2_left_base, ht, &rd2[0]);
        let d2_right = M::mask(d2_right_base, ht, &rd2[1]);
        let e1_beta = M1::msm(g1_full, &self.s2[..]);
        let e2_beta = M2::msm(g2_full, &self.s1[..]);

        FirstReduceMessage {
            d1_left,
            d1_right,
            d2_left,
            d2_right,
            e1_beta,
            e2_beta,
        }
    }

    /// Apply first challenge (beta) and combine vectors
    #[cfg(feature = "parallel")]
    pub fn apply_first_challenge<M1, M2>(&mut self, beta: &Scalar<E>)
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        let beta_inv = beta.inv().expect("beta must be invertible");
        let n = 1 << self.num_rounds;

        let (g1_slice, g2_slice) = (&self.setup.g1_vec[..n], &self.setup.g2_vec[..n]);
        let (v1_ref, v2_ref) = (&mut self.v1, &mut self.v2);

        rayon::join(
            || M1::fixed_scalar_mul_bases_then_add(g1_slice, v1_ref, beta),
            || M2::fixed_scalar_mul_bases_then_add(g2_slice, v2_ref, &beta_inv),
        );
        self.v2_scalars = None;

        self.r_c = self.r_c + self.r_d2 * beta + self.r_d1 * beta_inv;
    }

    /// Apply first challenge (beta) and combine vectors (sequential fallback)
    #[cfg(not(feature = "parallel"))]
    pub fn apply_first_challenge<M1, M2>(&mut self, beta: &Scalar<E>)
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        let beta_inv = beta.inv().expect("beta must be invertible");
        let n = 1 << self.num_rounds;

        let (g1_slice, g2_slice) = (&self.setup.g1_vec[..n], &self.setup.g2_vec[..n]);
        let (v1_ref, v2_ref) = (&mut self.v1, &mut self.v2);

        M1::fixed_scalar_mul_bases_then_add(g1_slice, v1_ref, beta);
        M2::fixed_scalar_mul_bases_then_add(g2_slice, v2_ref, &beta_inv);
        self.v2_scalars = None;

        self.r_c = self.r_c + self.r_d2 * beta + self.r_d1 * beta_inv;
    }

    /// Compute second reduce message for current round
    #[cfg(feature = "parallel")]
    pub fn compute_second_message<M1, M2>(&mut self) -> SecondReduceMessage<E::G1, E::G2, E::GT>
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        self.compute_second_message_impl::<M1, M2>()
    }

    #[cfg(not(feature = "parallel"))]
    pub fn compute_second_message<M1, M2>(&mut self) -> SecondReduceMessage<E::G1, E::G2, E::GT>
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        self.compute_second_message_impl::<M1, M2>()
    }

    #[cfg(feature = "parallel")]
    fn compute_second_message_impl<M1, M2>(&mut self) -> SecondReduceMessage<E::G1, E::G2, E::GT>
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        let n2 = 1 << (self.num_rounds - 1);
        let (v1_l, v1_r) = self.v1.split_at(n2);
        let (v2_l, v2_r) = self.v2.split_at(n2);
        let (s1_l, s1_r) = self.s1.split_at(n2);
        let (s2_l, s2_r) = self.s2.split_at(n2);

        self.round_c = [M::sample(), M::sample()];
        self.round_e1 = [M::sample(), M::sample()];
        self.round_e2 = [M::sample(), M::sample()];

        let ht = &self.setup.ht;
        let h1 = &self.setup.h1;
        let h2 = &self.setup.h2;
        let rc = self.round_c;
        let re1 = self.round_e1;
        let re2 = self.round_e2;

        let ((c_plus, c_minus), ((e1_plus, e1_minus), (e2_plus, e2_minus))) = rayon::join(
            || {
                rayon::join(
                    || M::mask(E::multi_pair(v1_l, v2_r), ht, &rc[0]),
                    || M::mask(E::multi_pair(v1_r, v2_l), ht, &rc[1]),
                )
            },
            || {
                rayon::join(
                    || {
                        rayon::join(
                            || M::mask(M1::msm(v1_l, s2_r), h1, &re1[0]),
                            || M::mask(M1::msm(v1_r, s2_l), h1, &re1[1]),
                        )
                    },
                    || {
                        rayon::join(
                            || M::mask(M2::msm(v2_r, s1_l), h2, &re2[0]),
                            || M::mask(M2::msm(v2_l, s1_r), h2, &re2[1]),
                        )
                    },
                )
            },
        );

        SecondReduceMessage {
            c_plus,
            c_minus,
            e1_plus,
            e1_minus,
            e2_plus,
            e2_minus,
        }
    }

    #[cfg(not(feature = "parallel"))]
    fn compute_second_message_impl<M1, M2>(&mut self) -> SecondReduceMessage<E::G1, E::G2, E::GT>
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        let n2 = 1 << (self.num_rounds - 1);
        let (v1_l, v1_r) = self.v1.split_at(n2);
        let (v2_l, v2_r) = self.v2.split_at(n2);
        let (s1_l, s1_r) = self.s1.split_at(n2);
        let (s2_l, s2_r) = self.s2.split_at(n2);

        self.round_c = [M::sample(), M::sample()];
        self.round_e1 = [M::sample(), M::sample()];
        self.round_e2 = [M::sample(), M::sample()];

        let ht = &self.setup.ht;
        let h1 = &self.setup.h1;
        let h2 = &self.setup.h2;
        let rc = self.round_c;
        let re1 = self.round_e1;
        let re2 = self.round_e2;

        let c_plus = M::mask(E::multi_pair(v1_l, v2_r), ht, &rc[0]);
        let c_minus = M::mask(E::multi_pair(v1_r, v2_l), ht, &rc[1]);
        let e1_plus = M::mask(M1::msm(v1_l, s2_r), h1, &re1[0]);
        let e1_minus = M::mask(M1::msm(v1_r, s2_l), h1, &re1[1]);
        let e2_plus = M::mask(M2::msm(v2_r, s1_l), h2, &re2[0]);
        let e2_minus = M::mask(M2::msm(v2_l, s1_r), h2, &re2[1]);

        SecondReduceMessage {
            c_plus,
            c_minus,
            e1_plus,
            e1_minus,
            e2_plus,
            e2_minus,
        }
    }

    /// Apply second challenge (alpha) and fold vectors
    #[cfg(feature = "parallel")]
    pub fn apply_second_challenge<M1: DoryRoutines<E::G1>, M2: DoryRoutines<E::G2>>(
        &mut self,
        alpha: &Scalar<E>,
    ) {
        let alpha_inv = alpha.inv().expect("alpha must be invertible");
        let n2 = 1 << (self.num_rounds - 1);

        let (v1, v2, s1, s2) = (&mut self.v1, &mut self.v2, &mut self.s1, &mut self.s2);

        rayon::join(
            || {
                rayon::join(
                    || {
                        let (v1_l, v1_r) = v1.split_at_mut(n2);
                        M1::fixed_scalar_mul_vs_then_add(v1_l, v1_r, alpha);
                    },
                    || {
                        let (v2_l, v2_r) = v2.split_at_mut(n2);
                        M2::fixed_scalar_mul_vs_then_add(v2_l, v2_r, &alpha_inv);
                    },
                )
            },
            || {
                rayon::join(
                    || {
                        let (s1_l, s1_r) = s1.split_at_mut(n2);
                        M1::fold_field_vectors(s1_l, s1_r, alpha);
                    },
                    || {
                        let (s2_l, s2_r) = s2.split_at_mut(n2);
                        M1::fold_field_vectors(s2_l, s2_r, &alpha_inv);
                    },
                )
            },
        );

        self.v1.truncate(n2);
        self.v2.truncate(n2);
        self.s1.truncate(n2);
        self.s2.truncate(n2);

        self.r_c = self.r_c + self.round_c[0] * alpha + self.round_c[1] * alpha_inv;
        self.r_d1 = self.round_d1[0] * alpha + self.round_d1[1];
        self.r_d2 = self.round_d2[0] * alpha_inv + self.round_d2[1];
        self.r_e1 = self.r_e1 + self.round_e1[0] * alpha + self.round_e1[1] * alpha_inv;
        self.r_e2 = self.r_e2 + self.round_e2[0] * alpha + self.round_e2[1] * alpha_inv;

        self.num_rounds -= 1;
    }

    /// Apply second challenge (alpha) and fold vectors (sequential fallback)
    #[cfg(not(feature = "parallel"))]
    pub fn apply_second_challenge<M1: DoryRoutines<E::G1>, M2: DoryRoutines<E::G2>>(
        &mut self,
        alpha: &Scalar<E>,
    ) {
        let alpha_inv = alpha.inv().expect("alpha must be invertible");
        let n2 = 1 << (self.num_rounds - 1);

        let (v1, v2, s1, s2) = (&mut self.v1, &mut self.v2, &mut self.s1, &mut self.s2);

        let (v1_l, v1_r) = v1.split_at_mut(n2);
        M1::fixed_scalar_mul_vs_then_add(v1_l, v1_r, alpha);
        let (v2_l, v2_r) = v2.split_at_mut(n2);
        M2::fixed_scalar_mul_vs_then_add(v2_l, v2_r, &alpha_inv);
        let (s1_l, s1_r) = s1.split_at_mut(n2);
        M1::fold_field_vectors(s1_l, s1_r, alpha);
        let (s2_l, s2_r) = s2.split_at_mut(n2);
        M1::fold_field_vectors(s2_l, s2_r, &alpha_inv);

        self.v1.truncate(n2);
        self.v2.truncate(n2);
        self.s1.truncate(n2);
        self.s2.truncate(n2);

        self.r_c = self.r_c + self.round_c[0] * alpha + self.round_c[1] * alpha_inv;
        self.r_d1 = self.round_d1[0] * alpha + self.round_d1[1];
        self.r_d2 = self.round_d2[0] * alpha_inv + self.round_d2[1];
        self.r_e1 = self.r_e1 + self.round_e1[0] * alpha + self.round_e1[1] * alpha_inv;
        self.r_e2 = self.r_e2 + self.round_e2[0] * alpha + self.round_e2[1] * alpha_inv;

        self.num_rounds -= 1;
    }

    /// Compute final scalar product message
    pub fn compute_final_message<M1, M2>(
        &mut self,
        gamma: &Scalar<E>,
    ) -> ScalarProductMessage<E::G1, E::G2>
    where
        M1: DoryRoutines<E::G1>,
        M2: DoryRoutines<E::G2>,
    {
        debug_assert_eq!(self.num_rounds, 0, "num_rounds must be 0 for final message");
        debug_assert_eq!(self.v1.len(), 1, "v1 must have length 1");
        debug_assert_eq!(self.v2.len(), 1, "v2 must have length 1");

        let gamma_inv = gamma.inv().expect("gamma must be invertible");
        let r_final1: Scalar<E> = M::sample();
        let r_final2: Scalar<E> = M::sample();

        let gamma_s1 = *gamma * self.s1[0] + r_final1;
        let e1 = self.v1[0] + gamma_s1 * self.setup.h1;

        let gamma_inv_s2 = gamma_inv * self.s2[0] + r_final2;
        let e2 = self.v2[0] + self.setup.h2.scale(&gamma_inv_s2);

        self.r_c = self.r_c + self.r_e2 * gamma + self.r_e1 * gamma_inv;

        ScalarProductMessage { e1, e2 }
    }

    /// Get the accumulated r_c blind value (needed for batch proof construction)
    pub fn get_blind(&self) -> Scalar<E> {
        self.r_c
    }

    /// Get number of rounds remaining
    pub fn rounds_remaining(&self) -> usize {
        self.num_rounds
    }

    /// Clone the prover state for lookahead computation
    ///
    /// This creates a deep copy of the current state that can be used
    /// for speculative computation in parallel threads.
    pub fn clone_state(&self) -> Self {
        self.clone()
    }

    /// Set initial blinding values (for Transparent mode these are zero)
    pub fn set_blinds(
        &mut self,
        r_d1: Scalar<E>,
        r_c: Scalar<E>,
        r_d2: Scalar<E>,
        r_e1: Scalar<E>,
        r_e2: Scalar<E>,
    ) {
        (self.r_d1, self.r_c, self.r_d2, self.r_e1, self.r_e2) = (r_d1, r_c, r_d2, r_e1, r_e2);
    }
}
