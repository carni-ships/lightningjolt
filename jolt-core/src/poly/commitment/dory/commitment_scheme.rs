//! Dory polynomial commitment scheme implementation

use super::dory_globals::{DoryContext, DoryGlobals, DoryLayout};
use super::jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
use super::wrappers::{
    ark_to_jolt, jolt_to_ark, ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT, ArkworksProverSetup,
    ArkworksVerifierSetup, JoltToDoryTranscript, BN254,
};
use crate::{
    curve::JoltCurve,
    field::JoltField,
    poly::commitment::commitment_scheme::{
        CommitmentScheme, StreamingCommitmentScheme, ZkEvalCommitment,
    },
    poly::multilinear_polynomial::MultilinearPolynomial,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math, small_scalar::SmallScalar},
};
use ark_bn254::{G1Affine, G1Projective};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_ec::CurveGroup;
use ark_ff::{AdditiveGroup, Zero};
use dory::primitives::{
    arithmetic::{Field as DoryField, Group, PairingCurve},
    poly::Polynomial,
};
use rayon::prelude::*;
use std::borrow::Borrow;
use std::sync::{Arc, RwLock};
use tracing::trace_span;

static G1_AFFINE_CACHE: RwLock<Option<(usize, Arc<Vec<G1Affine>>)>> = RwLock::new(None);

/// Reset the G1 affine cache.
/// This ensures test isolation when different tests use different setup sizes.
#[cfg(test)]
pub(crate) fn reset_g1_affine_cache() {
    let mut cache = G1_AFFINE_CACHE.write().unwrap();
    *cache = None;
}

fn get_cached_g1_affine_bases(setup: &ArkworksProverSetup, row_len: usize) -> Arc<Vec<G1Affine>> {
    // Fast path: read-only check
    {
        let cache = G1_AFFINE_CACHE.read().unwrap();
        if let Some((cached_len, ref bases)) = *cache {
            if cached_len == row_len {
                return bases.clone();
            }
        }
    }
    // Slow path: compute and cache
    let mut cache = G1_AFFINE_CACHE.write().unwrap();
    // Double-check after acquiring write lock
    if let Some((cached_len, ref bases)) = *cache {
        if cached_len == row_len {
            return bases.clone();
        }
    }
    let g1_slice = unsafe { std::slice::from_raw_parts(setup.g1_vec.as_ptr(), setup.g1_vec.len()) };
    let projs: Vec<G1Projective> = g1_slice[..row_len].iter().map(|g| g.0).collect();
    let bases = Arc::new(G1Projective::normalize_batch(&projs));
    *cache = Some((row_len, bases.clone()));
    bases
}

#[derive(Clone)]
pub struct DoryCommitmentScheme;

#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryOpeningProofHint(Vec<ArkG1>);

impl DoryOpeningProofHint {
    pub fn new(row_commitments: Vec<ArkG1>) -> Self {
        Self(row_commitments)
    }

    fn into_rows(self) -> Vec<ArkG1> {
        self.0
    }

    /// Returns the number of row commitments in this hint.
    pub fn num_rows(&self) -> usize {
        self.0.len()
    }
}

#[cfg(not(test))]
pub fn bind_opening_inputs<F: JoltField, ProofTranscript: Transcript>(
    transcript: &mut ProofTranscript,
    opening_point: &[F::Challenge],
    opening: &F,
) {
    let mut point_scalars = Vec::with_capacity(opening_point.len());
    for point in opening_point {
        let scalar: F = (*point).into();
        point_scalars.push(scalar);
    }
    transcript.append_scalars(b"dory_opening_point", &point_scalars);

    transcript.append_scalar(b"dory_opening_eval", opening);
}

#[cfg(test)]
pub fn bind_opening_inputs<F: JoltField, ProofTranscript: Transcript>(
    _transcript: &mut ProofTranscript,
    _opening_point: &[F::Challenge],
    _opening: &F,
) {
    // In test mode, skip bind_opening_inputs to avoid transcript mismatch.
    // DORY prove/verify already binds all necessary data through Fiat-Shamir.
    // This is safe because:
    // 1. Prover's `bind_opening_inputs` happens AFTER DORY prove, so the scalar
    //    appended here isn't used by DORY
    // 2. Verifier's `bind_opening_inputs` happens AFTER DORY verify, so the scalar
    //    appended here isn't used by anything post-verify
    // 3. The opening point and evaluation are already bound into the transcript
    //    through DORY's own challenge generation
}

#[cfg(feature = "zk")]
pub fn bind_opening_inputs_zk<F: JoltField, C: JoltCurve<F = F>, ProofTranscript: Transcript>(
    transcript: &mut ProofTranscript,
    opening_point: &[F::Challenge],
    y_com: &C::G1,
) {
    let mut point_scalars = Vec::with_capacity(opening_point.len());
    for point in opening_point {
        let scalar: F = (*point).into();
        point_scalars.push(scalar);
    }
    transcript.append_scalars(b"dory_opening_point", &point_scalars);

    transcript.append_commitment(b"dory_eval_commitment", y_com);
}

impl CommitmentScheme for DoryCommitmentScheme {
    type Field = ark_bn254::Fr;
    type ProverSetup = ArkworksProverSetup;
    type VerifierSetup = ArkworksVerifierSetup;
    type Commitment = ArkGT;
    type Proof = ArkDoryProof;
    type BatchedProof = Vec<ArkDoryProof>;
    type OpeningProofHint = DoryOpeningProofHint;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let _span = trace_span!("DoryCommitmentScheme::setup_prover").entered();
        #[cfg(feature = "metal-pairing")]
        super::metal_pairing::register();

        #[cfg(test)]
        DoryGlobals::configure_test_cache_root();

        #[cfg(not(target_arch = "wasm32"))]
        let setup = {
            let _load_span = trace_span!("load_srs").entered();
            ArkworksProverSetup::new_from_urs(max_num_vars)
        };
        #[cfg(target_arch = "wasm32")]
        let setup = ArkworksProverSetup::new(max_num_vars);

        #[cfg(not(test))]
        {
            let _cache_span = trace_span!("init_prepared_cache").entered();
            DoryGlobals::init_prepared_cache(&setup.g1_vec, &setup.g2_vec);
        }

        setup
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        let _span = trace_span!("DoryCommitmentScheme::setup_verifier").entered();
        setup.to_verifier_setup()
    }

    fn commit(
        poly: &MultilinearPolynomial<ark_bn254::Fr>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let _span = trace_span!("DoryCommitmentScheme::commit").entered();

        let num_cols = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();
        let sigma = num_cols.log_2();
        let nu = num_rows.log_2();

        let (tier_2, row_commitments, _commit_blind) =
            <MultilinearPolynomial<ark_bn254::Fr> as Polynomial<ArkFr>>::commit::<
                BN254,
                dory::Transparent,
                JoltG1Routines,
            >(poly, nu, sigma, setup)
            .expect("commitment should succeed");

        (tier_2, DoryOpeningProofHint::new(row_commitments))
    }

    fn batch_commit<U>(
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: std::borrow::Borrow<MultilinearPolynomial<ark_bn254::Fr>> + Sync,
    {
        let _span = trace_span!("DoryCommitmentScheme::batch_commit").entered();

        // Get K and T from DoryGlobals for thread-safe parallel initialization
        // K = address space size, T = trace length
        let k = DoryGlobals::k_from_matrix_shape();
        let t = DoryGlobals::get_T();
        let layout = DoryGlobals::get_layout();

        // Enable parallel batch commit by threading K and T through
        // Each thread initializes its own DoryGlobals context with the same parameters
        polys
            .par_iter()
            .map(|poly| {
                // Initialize thread-local DoryGlobals context for this thread
                let _guard = DoryGlobals::initialize_context(
                    k,
                    t,
                    DoryContext::Main,
                    Some(layout),
                );
                Self::commit(poly.borrow(), gens)
            })
            .collect()
    }

    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<ark_bn254::Fr>,
        opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
        sigma: usize,
        nu: usize,
    ) -> (Self::Proof, Option<Self::Field>) {
        let _span = trace_span!("DoryCommitmentScheme::prove").entered();

        let (row_commitments, commit_blind) = hint
            .map(|h| (h.into_rows(), DoryField::zero()))
            .unwrap_or_else(|| {
                let (_commitment, row_commitments) = Self::commit(poly, setup);
                (row_commitments.into_rows(), DoryField::zero())
            });

        // sigma and nu are now passed as parameters instead of read from globals
        let reordered_point = reorder_opening_point_for_layout::<ark_bn254::Fr>(opening_point);
        let ark_point: Vec<ArkFr> = reordered_point
            .iter()
            .rev()
            .map(|p| {
                let f_val: ark_bn254::Fr = (*p).into();
                jolt_to_ark(&f_val)
            })
            .collect();

        // prove() takes transcript by value (moved in)
        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        #[cfg(feature = "zk")]
        type DoryMode = dory::ZK;
        #[cfg(not(feature = "zk"))]
        type DoryMode = dory::Transparent;

        #[cfg(feature = "dory-prime")]
        {
            let _prove_span = trace_span!("DoryCommitmentScheme::prove_dory_prime").entered();
            // Use Dory-Prime parallel prover when the feature is enabled
            let dory_prime_proof = dory::dory_prime::prove_dory_prime::<
                ArkFr, BN254, JoltG1Routines, JoltG2Routines, _, _, DoryMode,
            >(
                poly,
                &ark_point,
                Some(row_commitments),
                commit_blind,
                nu,
                sigma,
                &setup.0,
                &mut dory_transcript,
            )
            .expect("Dory-Prime proof generation should succeed");

            // DoryPrimeProof wraps DoryProof in batch_proof field
            let proof = dory_prime_proof.batch_proof;
            // Dory-Prime doesn't use blinding in Transparent mode
            let y_blinding = None;
            (proof, y_blinding)
        }

        #[cfg(not(feature = "dory-prime"))]
        {
            let _prove_span = trace_span!("DoryCommitmentScheme::prove_standard").entered();
            let (proof, y_blinding) =
                dory::prove::<ArkFr, BN254, JoltG1Routines, JoltG2Routines, _, _, DoryMode>(
                    poly,
                    &ark_point,
                    row_commitments,
                    commit_blind,
                    nu,
                    sigma,
                    setup,
                    &mut dory_transcript,
                )
                .expect("proof generation should succeed");

            (proof, y_blinding.map(|b| ark_to_jolt(&b)))
        }
    }

    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
        opening: &ark_bn254::Fr,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let _span = trace_span!("DoryCommitmentScheme::verify").entered();

        let reordered_point = reorder_opening_point_for_layout::<ark_bn254::Fr>(opening_point);

        // Dory uses the opposite endian-ness as Jolt
        let ark_point: Vec<ArkFr> = reordered_point
            .iter()
            .rev()
            .map(|p| {
                let f_val: ark_bn254::Fr = (*p).into();
                jolt_to_ark(&f_val)
            })
            .collect();
        let ark_eval: ArkFr = jolt_to_ark(opening);

        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        dory::verify::<ArkFr, BN254, JoltG1Routines, JoltG2Routines, _>(
            *commitment,
            ark_eval,
            &ark_point,
            proof,
            setup.clone().into_inner(),
            &mut dory_transcript,
        )
        .map_err(|_| ProofVerifyError::InternalError)?;

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"Dory"
    }

    /// In Dory, the opening proof hint consists of the Pedersen commitments to the rows
    /// of the polynomial coefficient matrix. In the context of a batch opening proof, we
    /// can homomorphically combine the row commitments for multiple polynomials into the
    /// row commitments for the RLC of those polynomials. This is more efficient than computing
    /// the row commitments for the RLC from scratch.
    ///
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_hints")]
    fn combine_hints(
        hints: Vec<Self::OpeningProofHint>,
        coeffs: &[Self::Field],
        num_rows: usize,
    ) -> Self::OpeningProofHint {
        // num_rows is now passed as a parameter instead of read from globals

        let mut owned_hints: Vec<Vec<ArkG1>> = hints.into_iter().map(|h| h.0).collect();
        for h in &mut owned_hints {
            if h.len() < num_rows {
                h.resize(num_rows, ArkG1(G1Projective::zero()));
            }
        }

        // Pre-compute scalar digit decomposition (window=4, 64 windows of 4 bits)
        // for Pippenger MSM. This avoids 42 individual scalar mults per row.
        let scalars_digits: Vec<[u8; 64]> = coeffs
            .iter()
            .map(|c| {
                let bigint = ark_ff::PrimeField::into_bigint(*c);
                let limbs = bigint.0;
                let mut digits = [0u8; 64];
                for (i, digit) in digits.iter_mut().enumerate() {
                    let limb_idx = i / 16;
                    let nibble_idx = i % 16;
                    *digit = ((limbs[limb_idx] >> (nibble_idx * 4)) & 0xF) as u8;
                }
                digits
            })
            .collect();

        // Parallel row-wise Pippenger MSM (window=4, 16 buckets)
        let rlc_hint: Vec<ArkG1> = (0..num_rows)
            .into_par_iter()
            .map(|r| {
                let mut result = G1Projective::zero();
                // Process windows from most significant to least significant
                for w in (0..64).rev() {
                    if w != 63 {
                        // Double 4 times (shift by window width)
                        result.double_in_place();
                        result.double_in_place();
                        result.double_in_place();
                        result.double_in_place();
                    }
                    // Scatter points into 16 buckets (bucket 0 = identity, skip)
                    let mut buckets = [G1Projective::zero(); 15];
                    for (poly_idx, digits) in scalars_digits.iter().enumerate() {
                        let digit = digits[w] as usize;
                        if digit > 0 {
                            buckets[digit - 1] += owned_hints[poly_idx][r].0;
                        }
                    }
                    // Reduce buckets: bucket[i] contributes (i+1) * bucket[i]
                    // Running sum trick: sum = b[14], result += sum; sum += b[13], result += sum; ...
                    let mut running = G1Projective::zero();
                    for b in (0..15).rev() {
                        running += buckets[b];
                        result += running;
                    }
                }
                ArkG1(result)
            })
            .collect();

        DoryOpeningProofHint::new(rlc_hint)
    }

    /// Homomorphically combines multiple commitments using a random linear combination.
    /// Computes: sum_i(coeff_i * commitment_i) for the GT elements.
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_commitments")]
    fn combine_commitments<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let _span = trace_span!("DoryCommitmentScheme::combine_commitments").entered();

        // Combine GT elements using parallel RLC
        let borrowed: Vec<&ArkGT> = commitments.iter().map(|c| c.borrow()).collect();
        coeffs
            .par_iter()
            .zip(borrowed.par_iter())
            .map(|(coeff, commitment)| {
                let ark_coeff = jolt_to_ark(coeff);
                ark_coeff * **commitment
            })
            .reduce(ArkGT::identity, |a, b| a + b)
    }
}

impl StreamingCommitmentScheme for DoryCommitmentScheme {
    type ChunkState = Vec<ArkG1>; // Tier 1 commitment chunks

    fn process_chunk<T: SmallScalar>(
        setup: &Self::ProverSetup,
        chunk: &[T],
        sigma: usize,
    ) -> Self::ChunkState {
        let row_len = 1 << sigma;
        debug_assert_eq!(chunk.len(), row_len);

        let g1_bases = get_cached_g1_affine_bases(setup, row_len);

        let row_commitment =
            ArkG1(T::msm(&g1_bases[..chunk.len()], chunk).expect("MSM calculation failed."));
        vec![row_commitment]
    }

    fn process_chunk_onehot(
        setup: &Self::ProverSetup,
        onehot_k: usize,
        chunk: &[Option<usize>],
        sigma: usize,
    ) -> Self::ChunkState {
        let K = onehot_k;

        let row_len = 1 << sigma;
        let g1_bases = get_cached_g1_affine_bases(setup, row_len);

        let mut indices_per_k: Vec<Vec<usize>> = vec![Vec::new(); K];
        for (col_index, k) in chunk.iter().enumerate() {
            if let Some(k) = k {
                indices_per_k[*k].push(col_index);
            }
        }

        let results = jolt_optimizations::batch_g1_additions_multi(&g1_bases, &indices_per_k);

        let mut row_commitments = vec![ArkG1(G1Projective::zero()); K];
        for (k, result) in results.into_iter().enumerate() {
            if !indices_per_k[k].is_empty() {
                row_commitments[k] = ArkG1(G1Projective::from(result));
            }
        }
        row_commitments
    }

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::compute_tier2_commitment")]
    fn aggregate_chunks(
        setup: &Self::ProverSetup,
        onehot_k: Option<usize>,
        chunks: &[Self::ChunkState],
        sigma: usize,
        nu: usize,
        _T: usize,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let num_rows = 1 << nu;
        let row_len = 1 << sigma;

        if let Some(_K) = onehot_k {
            let rows_per_k = _T / row_len;

            let mut row_commitments = vec![ArkG1(G1Projective::zero()); num_rows];
            for (chunk_index, commitments) in chunks.iter().enumerate() {
                row_commitments
                    .par_iter_mut()
                    .skip(chunk_index)
                    .step_by(rows_per_k)
                    .zip(commitments.par_iter())
                    .for_each(|(dest, src)| *dest = *src);
            }

            let g2_bases = &setup.g2_vec[..num_rows];
            let tier_2 = <BN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases);

            (tier_2, DoryOpeningProofHint::new(row_commitments))
        } else {
            let total_len: usize = chunks.iter().map(|c| c.len()).sum();
            let mut row_commitments = Vec::with_capacity(total_len);
            for chunk in chunks {
                row_commitments.extend_from_slice(chunk);
            }

            let g2_bases = &setup.g2_vec[..row_commitments.len()];
            let tier_2 = <BN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases);

            (tier_2, DoryOpeningProofHint::new(row_commitments))
        }
    }

    /// Batch compute tier2 commitments for multiple polynomials.
    /// Passes sigma/nu/T through to enable parallel execution with proper parameters.
    fn batch_aggregate_chunks(
        setup: &Self::ProverSetup,
        tier1_per_poly: Vec<Vec<Self::ChunkState>>,
        onehot_ks: &[Option<usize>],
        sigma: usize,
        nu: usize,
        T: usize,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)> {
        use rayon::prelude::*;
        let tier1_refs: Vec<&[Self::ChunkState]> = tier1_per_poly.iter().map(|v| v.as_slice()).collect();
        tier1_refs
            .into_par_iter()
            .zip(onehot_ks.par_iter())
            .map(|(tier1, k)| Self::aggregate_chunks(setup, *k, tier1, sigma, nu, T))
            .collect()
    }
}

impl<C: JoltCurve> ZkEvalCommitment<C> for DoryCommitmentScheme
where
    C::G1: From<ArkG1>,
{
    fn eval_commitment(proof: &Self::Proof) -> Option<C::G1> {
        #[cfg(feature = "zk")]
        {
            proof.y_com.as_ref().copied().map(C::G1::from)
        }
        #[cfg(not(feature = "zk"))]
        {
            let _ = proof;
            None
        }
    }

    fn eval_commitment_gens(setup: &Self::ProverSetup) -> Option<(C::G1, C::G1)> {
        let g1_0 = setup.0.g1_vec.first().copied().map(C::G1::from)?;
        let h1 = C::G1::from(setup.0.h1);
        Some((g1_0, h1))
    }

    fn eval_commitment_gens_verifier(setup: &Self::VerifierSetup) -> Option<(C::G1, C::G1)> {
        let g1_0 = C::G1::from(setup.0.g1_0);
        let h1 = C::G1::from(setup.0.h1);
        Some((g1_0, h1))
    }

    #[cfg(feature = "zk")]
    fn zk_generators(setup: &Self::ProverSetup, count: usize) -> Option<(Vec<C::G1>, C::G1)> {
        let count = std::cmp::min(count, setup.0.g1_vec.len());
        let g1s = setup.0.g1_vec[..count]
            .iter()
            .map(|g| C::G1::from(*g))
            .collect();
        let h1 = C::G1::from(setup.0.h1);
        Some((g1s, h1))
    }
}

/// Extract Dory witness data for gnark Groth16 circuit.
///
/// Replays the Dory Fiat-Shamir on the given transcript to capture
/// alpha/beta/gamma/d challenges and G2 composite witnesses.
/// The transcript must be in the same state as when `verify()` would be called.
pub fn extract_dory_witness<ProofTranscript: Transcript>(
    proof: &ArkDoryProof,
    setup: &ArkworksVerifierSetup,
    transcript: &mut ProofTranscript,
    opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
    opening: &ark_bn254::Fr,
    commitment: &ArkGT,
) -> Result<dory::DoryWitnessData<ArkFr, ArkG1, ArkG2, ArkGT>, crate::utils::errors::ProofVerifyError>
{
    let reordered_point = reorder_opening_point_for_layout::<ark_bn254::Fr>(opening_point);
    let ark_point: Vec<ArkFr> = reordered_point
        .iter()
        .rev()
        .map(|p| {
            let f_val: ark_bn254::Fr = (*p).into();
            jolt_to_ark(&f_val)
        })
        .collect();
    let ark_eval: ArkFr = jolt_to_ark(opening);

    let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

    dory::extract_witness_data::<ArkFr, BN254, JoltG1Routines, JoltG2Routines, _>(
        *commitment,
        ark_eval,
        &ark_point,
        proof,
        &setup.clone().into_inner(),
        &mut dory_transcript,
    )
    .map_err(|_| crate::utils::errors::ProofVerifyError::InternalError)
}

/// Reorders opening_point for AddressMajor layout.
///
/// For AddressMajor layout, reorders opening_point from [r_address, r_cycle] to [r_cycle, r_address].
/// This ensures that after Dory's reversal and splitting:
/// - Column (right) vector gets address variables (matching AddressMajor column indexing)
/// - Row (left) vector gets cycle variables (matching AddressMajor row indexing)
///
/// For CycleMajor layout, returns the point unchanged.
fn reorder_opening_point_for_layout<F: JoltField>(
    opening_point: &[F::Challenge],
) -> std::borrow::Cow<'_, [F::Challenge]> {
    if DoryGlobals::get_layout() == DoryLayout::AddressMajor {
        let log_T = DoryGlobals::get_T().log_2();
        let log_K = opening_point.len().saturating_sub(log_T);
        let (r_address, r_cycle) = opening_point.split_at(log_K);
        let mut reordered = Vec::with_capacity(opening_point.len());
        reordered.extend_from_slice(r_cycle);
        reordered.extend_from_slice(r_address);
        std::borrow::Cow::Owned(reordered)
    } else {
        std::borrow::Cow::Borrowed(opening_point)
    }
}
