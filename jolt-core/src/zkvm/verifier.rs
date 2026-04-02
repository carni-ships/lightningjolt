use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use crate::curve::JoltCurve;
use crate::poly::commitment::commitment_scheme::{CommitmentScheme, ZkEvalCommitment};
#[cfg(feature = "zk")]
use crate::poly::commitment::dory::bind_opening_inputs_zk;
use crate::poly::commitment::dory::{bind_opening_inputs, DoryContext, DoryGlobals};
use crate::poly::commitment::pedersen::PedersenGenerators;
#[cfg(feature = "zk")]
use crate::poly::lagrange_poly::LagrangeHelper;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{
    pedersen_generator_count_for_r1cs, BakedPublicInputs, BlindFoldVerifier,
    BlindFoldVerifierInput, ClaimBindingConfig, InputClaimConstraint, OutputClaimConstraint,
    StageConfig, ValueSource, VerifierR1CSBuilder,
};
use crate::subprotocols::sumcheck::BatchedSumcheck;
#[cfg(feature = "zk")]
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
#[cfg(feature = "zk")]
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
#[cfg(feature = "zk")]
use crate::subprotocols::univariate_skip::UniSkipFirstRoundProofVariant;
use crate::zkvm::bytecode::{BytecodePreprocessing, PreprocessingError};
use crate::zkvm::claim_reductions::advice::ReductionPhase;
use crate::zkvm::claim_reductions::RegistersClaimReductionSumcheckVerifier;
use crate::zkvm::config::OneHotParams;
#[cfg(feature = "prover")]
use crate::zkvm::prover::JoltProverPreprocessing;
#[cfg(feature = "zk")]
use crate::zkvm::r1cs::constraints::{
    OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
    PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
};
use crate::zkvm::ram::RAMPreprocessing;
use crate::zkvm::witness::all_committed_polynomials;
use crate::zkvm::Serializable;
use crate::zkvm::{
    bytecode::read_raf_checking::BytecodeReadRafSumcheckVerifier,
    claim_reductions::{
        AdviceClaimReductionVerifier, AdviceKind, HammingWeightClaimReductionVerifier,
        IncClaimReductionSumcheckVerifier, InstructionLookupsClaimReductionSumcheckVerifier,
        RamRaClaimReductionSumcheckVerifier,
    },
    fiat_shamir_preamble,
    instruction_lookups::{
        ra_virtual::RaSumcheckVerifier as LookupsRaSumcheckVerifier,
        read_raf_checking::InstructionReadRafSumcheckVerifier,
    },
    proof_serialization::JoltProof,
    r1cs::key::UniformSpartanKey,
    ram::{
        compute_min_ram_K, hamming_booleanity::HammingBooleanitySumcheckVerifier,
        output_check::OutputSumcheckVerifier, ra_virtual::RamRaVirtualSumcheckVerifier,
        raf_evaluation::RafEvaluationSumcheckVerifier as RamRafEvaluationSumcheckVerifier,
        read_write_checking::RamReadWriteCheckingVerifier, val_check::RamValCheckSumcheckVerifier,
        verifier_accumulate_advice,
    },
    registers::{
        read_write_checking::RegistersReadWriteCheckingVerifier,
        val_evaluation::ValEvaluationSumcheckVerifier as RegistersValEvaluationSumcheckVerifier,
    },
    spartan::{
        instruction_input::InstructionInputSumcheckVerifier, outer::OuterRemainingSumcheckVerifier,
        product::ProductVirtualRemainderVerifier, shift::ShiftSumcheckVerifier,
        verify_stage1_uni_skip, verify_stage2_uni_skip,
    },
    stage8_opening_ids, ProverDebugInfo,
};
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        opening_proof::{
            compute_advice_lagrange_factor, DoryOpeningState, OpeningAccumulator, OpeningId,
            SumcheckId, VerifierOpeningAccumulator,
        },
    },
    pprof_scope,
    subprotocols::{
        booleanity::{BooleanitySumcheckParams, BooleanitySumcheckVerifier},
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::{HasState, Transcript},
    utils::{errors::ProofVerifyError, math::Math},
    zkvm::witness::CommittedPolynomial,
};

#[cfg(not(feature = "zk"))]
use crate::zkvm::onchain_export::OnChainExportData;

#[cfg(feature = "zk")]
struct StageVerifyResult<F: JoltField> {
    challenges: Vec<F::Challenge>,
    batched_output_constraint: Option<OutputClaimConstraint>,
    output_constraint_challenge_values: Vec<F>,
    batched_input_constraint: InputClaimConstraint,
    input_constraint_challenge_values: Vec<F>,
    uniskip_input_constraint: Option<InputClaimConstraint>,
    uniskip_input_constraint_challenge_values: Vec<F>,
    oc_block_ids: Vec<Vec<OpeningId>>,
}

#[cfg(not(feature = "zk"))]
struct StageVerifyResult<F: JoltField> {
    #[allow(dead_code)]
    challenges: Vec<F::Challenge>,
}

#[cfg(feature = "zk")]
impl<F: JoltField> StageVerifyResult<F> {
    fn new(
        challenges: Vec<F::Challenge>,
        batched_output_constraint: Option<OutputClaimConstraint>,
        output_constraint_challenge_values: Vec<F>,
        batched_input_constraint: InputClaimConstraint,
        input_constraint_challenge_values: Vec<F>,
        oc_block_ids: Vec<Vec<OpeningId>>,
    ) -> Self {
        Self {
            challenges,
            batched_output_constraint,
            output_constraint_challenge_values,
            batched_input_constraint,
            input_constraint_challenge_values,
            uniskip_input_constraint: None,
            uniskip_input_constraint_challenge_values: Vec::new(),
            oc_block_ids,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn with_uniskip(
        challenges: Vec<F::Challenge>,
        batched_output_constraint: Option<OutputClaimConstraint>,
        output_constraint_challenge_values: Vec<F>,
        batched_input_constraint: InputClaimConstraint,
        input_constraint_challenge_values: Vec<F>,
        uniskip_input_constraint: InputClaimConstraint,
        uniskip_input_constraint_challenge_values: Vec<F>,
        oc_block_ids: Vec<Vec<OpeningId>>,
    ) -> Self {
        Self {
            challenges,
            batched_output_constraint,
            output_constraint_challenge_values,
            batched_input_constraint,
            input_constraint_challenge_values,
            uniskip_input_constraint: Some(uniskip_input_constraint),
            uniskip_input_constraint_challenge_values,
            oc_block_ids,
        }
    }
}

#[cfg(feature = "zk")]
fn batch_output_constraints<F: JoltField, T: Transcript>(
    instances: &[&dyn SumcheckInstanceVerifier<F, T>],
) -> Option<OutputClaimConstraint> {
    let constraints: Vec<Option<OutputClaimConstraint>> = instances
        .iter()
        .map(|instance| instance.get_params().output_claim_constraint())
        .collect();
    OutputClaimConstraint::batch(&constraints)
}

#[cfg(feature = "zk")]
fn batch_input_constraints<F: JoltField, T: Transcript>(
    instances: &[&dyn SumcheckInstanceVerifier<F, T>],
) -> InputClaimConstraint {
    let constraints: Vec<InputClaimConstraint> = instances
        .iter()
        .map(|instance| instance.get_params().input_claim_constraint())
        .collect();
    InputClaimConstraint::batch_required(&constraints, instances.len())
}

#[cfg(feature = "zk")]
fn scale_batching_coefficients<F: JoltField, T: Transcript>(
    batching_coefficients: &[F],
    instances: &[&dyn SumcheckInstanceVerifier<F, T>],
) -> Vec<F> {
    let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap_or(0);
    batching_coefficients
        .iter()
        .zip(instances.iter())
        .map(|(coeff, instance)| {
            let scale = max_num_rounds - instance.num_rounds();
            coeff.mul_pow_2(scale)
        })
        .collect()
}
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use tracer::instruction::Instruction;
use tracer::JoltDevice;

pub struct JoltVerifier<
    'a,
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
> {
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub program_io: JoltDevice,
    pub proof: JoltProof<F, C, PCS, ProofTranscript>,
    pub preprocessing: &'a JoltVerifierPreprocessing<F, C, PCS>,
    pub transcript: ProofTranscript,
    pub opening_accumulator: VerifierOpeningAccumulator<F>,
    /// The advice claim reduction sumcheck effectively spans two stages (6 and 7).
    /// Cache the verifier state here between stages.
    advice_reduction_verifier_trusted: Option<AdviceClaimReductionVerifier<F>>,
    /// The advice claim reduction sumcheck effectively spans two stages (6 and 7).
    /// Cache the verifier state here between stages.
    advice_reduction_verifier_untrusted: Option<AdviceClaimReductionVerifier<F>>,
    pub spartan_key: UniformSpartanKey<F>,
    pub one_hot_params: OneHotParams,
    /// Captured BytecodeReadRaf val_poly evaluations for on-chain export.
    #[cfg(not(feature = "zk"))]
    pub stage6_val_poly_evals: Option<[F; 5]>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct Stage8VerifyData<F: JoltField> {
    opening_ids: Vec<OpeningId>,
    constraint_coeffs: Vec<F>,
    /// Raw committed claims (before RLC), for on-chain export.
    claims: Vec<F>,
    /// Scaling factors (Lagrange factors for dense/advice, 1 for RA), for on-chain export.
    scaling_factors: Vec<F>,
    /// Unified opening point as field elements, for on-chain export.
    opening_point: Vec<F>,
    /// Unified opening point as challenge type, for Dory witness export.
    opening_point_challenges: Vec<F::Challenge>,
    /// Joint claim = Σ γ^i * claim_i, for on-chain export.
    joint_claim: F,
    /// Raw gamma powers for joint commitment reconstruction.
    gamma_powers: Vec<F>,
    /// (CommittedPolynomial, claim) pairs for joint commitment reconstruction.
    polynomial_claims: Vec<(CommittedPolynomial, F)>,
}

impl<
        'a,
        F: JoltField,
        C: JoltCurve<F = F>,
        PCS: CommitmentScheme<Field = F> + ZkEvalCommitment<C>,
        ProofTranscript: Transcript,
    > JoltVerifier<'a, F, C, PCS, ProofTranscript>
{
    pub fn new(
        preprocessing: &'a JoltVerifierPreprocessing<F, C, PCS>,
        proof: JoltProof<F, C, PCS, ProofTranscript>,
        mut program_io: JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        _debug_info: Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
    ) -> Result<Self, ProofVerifyError> {
        // Memory layout checks
        if program_io.memory_layout != preprocessing.shared.memory_layout {
            return Err(ProofVerifyError::MemoryLayoutMismatch);
        }
        if program_io.inputs.len() > preprocessing.shared.memory_layout.max_input_size as usize {
            return Err(ProofVerifyError::InputTooLarge);
        }
        if program_io.outputs.len() > preprocessing.shared.memory_layout.max_output_size as usize {
            return Err(ProofVerifyError::OutputTooLarge);
        }

        // truncate trailing zeros on device outputs
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        let zk_mode = proof.stage1_sumcheck_proof.is_zk();
        #[cfg(test)]
        #[allow(unused_mut)]
        let mut opening_accumulator =
            VerifierOpeningAccumulator::new(proof.trace_length.log_2(), zk_mode);
        #[cfg(not(test))]
        #[allow(unused_mut)]
        let mut opening_accumulator =
            VerifierOpeningAccumulator::new(proof.trace_length.log_2(), zk_mode);

        #[cfg(not(feature = "zk"))]
        {
            use crate::poly::opening_proof::{OpeningPoint, BIG_ENDIAN};
            for (id, (_, claim)) in &proof.opening_claims.0 {
                let dummy_point = OpeningPoint::<BIG_ENDIAN, F>::new(vec![]);
                opening_accumulator
                    .openings
                    .insert(*id, (dummy_point, *claim));
            }
        }

        #[cfg(test)]
        let mut transcript = ProofTranscript::new(b"Jolt");
        #[cfg(not(test))]
        let transcript = ProofTranscript::new(b"Jolt");

        #[cfg(test)]
        {
            if let Some(debug_info) = _debug_info {
                transcript.compare_to(debug_info.transcript);
                opening_accumulator.compare_to(debug_info.opening_accumulator);
            }
        }

        let spartan_key = UniformSpartanKey::new(proof.trace_length.next_power_of_two());

        // Validate configs from the proof
        proof
            .one_hot_config
            .validate()
            .map_err(ProofVerifyError::InvalidOneHotConfig)?;

        let min_ram_K = compute_min_ram_K(
            &preprocessing.shared.ram,
            &preprocessing.shared.memory_layout,
        );
        if !proof.ram_K.is_power_of_two() || proof.ram_K < min_ram_K {
            return Err(ProofVerifyError::InvalidRamK(proof.ram_K, min_ram_K));
        }

        proof
            .rw_config
            .validate(proof.trace_length.log_2(), proof.ram_K.log_2())
            .map_err(ProofVerifyError::InvalidReadWriteConfig)?;

        // Construct full params from the validated config.
        let bytecode_K = preprocessing.shared.bytecode.code_size;
        let one_hot_params =
            OneHotParams::from_config(&proof.one_hot_config, bytecode_K, proof.ram_K);

        Ok(Self {
            trusted_advice_commitment,
            program_io,
            proof,
            preprocessing,
            transcript,
            opening_accumulator,
            advice_reduction_verifier_trusted: None,
            advice_reduction_verifier_untrusted: None,
            spartan_key,
            one_hot_params,
            #[cfg(not(feature = "zk"))]
            stage6_val_poly_evals: None,
        })
    }

    /// Returns the transcript's raw state bytes (for export/testing).
    /// Only available when using KeccakTranscript (state is public).
    pub fn transcript_state_bytes(&self) -> &[u8; 32]
    where
        ProofTranscript: HasState,
    {
        self.transcript.state_bytes()
    }

    /// Returns the transcript's round counter (for export/testing).
    pub fn transcript_n_rounds(&self) -> u32
    where
        ProofTranscript: HasState,
    {
        self.transcript.n_rounds_value()
    }

    /// Export Stage 3 verification data for Solidity test vector generation.
    /// Returns (transcript_state_before_stage3, transcript_state_after_stage3, n_flushed_claims).
    #[cfg(test)]
    pub fn export_stage3_data(mut self) -> Result<([u8; 32], [u8; 32], usize), ProofVerifyError>
    where
        ProofTranscript: HasState,
    {
        fiat_shamir_preamble(
            &self.program_io,
            self.proof.ram_K,
            self.proof.trace_length,
            self.preprocessing.shared.bytecode.entry_address,
            &mut self.transcript,
        );
        for commitment in &self.proof.commitments {
            self.transcript
                .append_serializable(b"commitment", commitment);
        }
        if let Some(ref untrusted_advice_commitment) = self.proof.untrusted_advice_commitment {
            self.transcript
                .append_serializable(b"untrusted_advice", untrusted_advice_commitment);
        }
        if let Some(ref trusted_advice_commitment) = self.trusted_advice_commitment {
            self.transcript
                .append_serializable(b"trusted_advice", trusted_advice_commitment);
        }

        let _ = self.verify_stage1()?;
        let _ = self.verify_stage2()?;

        let state_before = *self.transcript.state_bytes();

        let _ = self.verify_stage3()?;

        let state_after = *self.transcript.state_bytes();
        let n_rounds_after = self.transcript.n_rounds_value();

        // Continue with remaining stages to fully verify the proof
        let _ = self.verify_stage4()?;
        let _ = self.verify_stage5()?;
        let _ = self.verify_stage6()?;
        let _ = self.verify_stage7()?;
        let _ = self.verify_stage8()?;

        Ok((state_before, state_after, n_rounds_after as usize))
    }

    #[tracing::instrument(skip_all)]
    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    pub fn verify(mut self) -> Result<(), ProofVerifyError> {
        let _pprof_verify = pprof_scope!("verify");
        let zk_mode = self.opening_accumulator.zk_mode;

        fiat_shamir_preamble(
            &self.program_io,
            self.proof.ram_K,
            self.proof.trace_length,
            self.preprocessing.shared.bytecode.entry_address,
            &mut self.transcript,
        );

        // Append commitments to transcript
        for commitment in &self.proof.commitments {
            self.transcript
                .append_serializable(b"commitment", commitment);
        }
        // Append untrusted advice commitment to transcript
        if let Some(ref untrusted_advice_commitment) = self.proof.untrusted_advice_commitment {
            self.transcript
                .append_serializable(b"untrusted_advice", untrusted_advice_commitment);
        }
        // Append trusted advice commitment to transcript
        if let Some(ref trusted_advice_commitment) = self.trusted_advice_commitment {
            self.transcript
                .append_serializable(b"trusted_advice", trusted_advice_commitment);
        }

        let (stage1_result, uniskip_challenge1) = self
            .verify_stage1()
            .inspect_err(|e| tracing::error!("Stage 1: {e}"))?;
        let (stage2_result, uniskip_challenge2) = self
            .verify_stage2()
            .inspect_err(|e| tracing::error!("Stage 2: {e}"))?;
        let stage3_result = self
            .verify_stage3()
            .inspect_err(|e| tracing::error!("Stage 3: {e}"))?;
        let stage4_result = self
            .verify_stage4()
            .inspect_err(|e| tracing::error!("Stage 4: {e}"))?;
        let stage5_result = self
            .verify_stage5()
            .inspect_err(|e| tracing::error!("Stage 5: {e}"))?;
        let stage6_result = self
            .verify_stage6()
            .inspect_err(|e| tracing::error!("Stage 6: {e}"))?;
        let stage7_result = self
            .verify_stage7()
            .inspect_err(|e| tracing::error!("Stage 7: {e}"))?;
        let stage8_data = self
            .verify_stage8()
            .inspect_err(|e| tracing::error!("Stage 8: {e}"))?;

        if zk_mode {
            #[cfg(feature = "zk")]
            {
                let sumcheck_challenges = [
                    stage1_result.challenges.clone(),
                    stage2_result.challenges.clone(),
                    stage3_result.challenges.clone(),
                    stage4_result.challenges.clone(),
                    stage5_result.challenges.clone(),
                    stage6_result.challenges.clone(),
                    stage7_result.challenges.clone(),
                ];
                let uniskip_challenges = [uniskip_challenge1, uniskip_challenge2];

                let stage_output_constraints = [
                    stage1_result.batched_output_constraint,
                    stage2_result.batched_output_constraint,
                    stage3_result.batched_output_constraint,
                    stage4_result.batched_output_constraint,
                    stage5_result.batched_output_constraint,
                    stage6_result.batched_output_constraint,
                    stage7_result.batched_output_constraint,
                ];

                let stage_input_constraints = [
                    stage1_result.uniskip_input_constraint.clone().unwrap(),
                    stage2_result.uniskip_input_constraint.clone().unwrap(),
                    stage3_result.batched_input_constraint.clone(),
                    stage4_result.batched_input_constraint.clone(),
                    stage5_result.batched_input_constraint.clone(),
                    stage6_result.batched_input_constraint.clone(),
                    stage7_result.batched_input_constraint.clone(),
                ];

                let stage_input_constraint_values = [
                    stage1_result
                        .uniskip_input_constraint_challenge_values
                        .clone(),
                    stage2_result
                        .uniskip_input_constraint_challenge_values
                        .clone(),
                    stage3_result.input_constraint_challenge_values.clone(),
                    stage4_result.input_constraint_challenge_values.clone(),
                    stage5_result.input_constraint_challenge_values.clone(),
                    stage6_result.input_constraint_challenge_values.clone(),
                    stage7_result.input_constraint_challenge_values.clone(),
                ];

                let output_constraint_challenge_values: [Vec<F>; 7] = [
                    stage1_result.output_constraint_challenge_values.clone(),
                    stage2_result.output_constraint_challenge_values.clone(),
                    stage3_result.output_constraint_challenge_values.clone(),
                    stage4_result.output_constraint_challenge_values.clone(),
                    stage5_result.output_constraint_challenge_values.clone(),
                    stage6_result.output_constraint_challenge_values.clone(),
                    stage7_result.output_constraint_challenge_values.clone(),
                ];

                let mut oc_blocks: Vec<Vec<OpeningId>> = Vec::new();
                oc_blocks.extend(stage1_result.oc_block_ids);
                oc_blocks.extend(stage2_result.oc_block_ids);
                oc_blocks.extend(stage3_result.oc_block_ids);
                oc_blocks.extend(stage4_result.oc_block_ids);
                oc_blocks.extend(stage5_result.oc_block_ids);
                oc_blocks.extend(stage6_result.oc_block_ids);
                oc_blocks.extend(stage7_result.oc_block_ids);

                self.verify_blindfold(
                    &sumcheck_challenges,
                    uniskip_challenges,
                    &stage_output_constraints,
                    &output_constraint_challenge_values,
                    &stage_input_constraints,
                    &stage_input_constraint_values,
                    &stage1_result.batched_input_constraint,
                    &stage2_result.batched_input_constraint,
                    &stage1_result.input_constraint_challenge_values,
                    &stage2_result.input_constraint_challenge_values,
                    &stage8_data,
                    oc_blocks,
                )?;
            }
            #[cfg(not(feature = "zk"))]
            return Err(ProofVerifyError::ZkFeatureRequired);
        }

        Ok(())
    }

    /// Run verification with flush recording enabled and return captured data.
    /// Only available in non-ZK mode.
    #[cfg(not(feature = "zk"))]
    pub fn verify_for_export(mut self) -> Result<OnChainExportData, ProofVerifyError>
    where
        ProofTranscript: crate::transcripts::HasState + Clone,
    {
        use crate::subprotocols::sumcheck::SumcheckInstanceProof;
        use crate::subprotocols::univariate_skip::UniSkipFirstRoundProofVariant;
        use crate::zkvm::onchain_export::{bytes_to_hex, extract_compressed_polys, fr_to_hex};

        self.opening_accumulator.enable_flush_recording();
        let mut transcript_states = Vec::new();
        let mut transcript_n_rounds = Vec::new();

        // Extract uni skip polynomial coefficients before verification
        let extract_uniskip_coeffs =
            |proof: &UniSkipFirstRoundProofVariant<F, C, ProofTranscript>| -> Vec<String> {
                match proof {
                    UniSkipFirstRoundProofVariant::Standard(p) => {
                        p.uni_poly.coeffs.iter().map(|c| fr_to_hex(c)).collect()
                    }
                    _ => vec![],
                }
            };
        let uniskip1_coeffs = extract_uniskip_coeffs(&self.proof.stage1_uni_skip_first_round_proof);
        let uniskip2_coeffs = extract_uniskip_coeffs(&self.proof.stage2_uni_skip_first_round_proof);

        // Extract compressed polynomials from each stage's sumcheck proof
        let extract_stage_polys =
            |proof: &SumcheckInstanceProof<F, C, ProofTranscript>| -> Vec<Vec<String>> {
                match proof {
                    SumcheckInstanceProof::Clear(clear) => extract_compressed_polys(clear),
                    _ => vec![],
                }
            };
        let stage_compressed_polys = vec![
            extract_stage_polys(&self.proof.stage1_sumcheck_proof),
            extract_stage_polys(&self.proof.stage2_sumcheck_proof),
            extract_stage_polys(&self.proof.stage3_sumcheck_proof),
            extract_stage_polys(&self.proof.stage4_sumcheck_proof),
            extract_stage_polys(&self.proof.stage5_sumcheck_proof),
            extract_stage_polys(&self.proof.stage6_sumcheck_proof),
            extract_stage_polys(&self.proof.stage7_sumcheck_proof),
        ];

        // Extract preamble data before it's consumed
        let trace_length = self.proof.trace_length;
        let ram_k = self.proof.ram_K;
        let entry_address = self.preprocessing.shared.bytecode.entry_address;
        // Must use program_io.memory_layout (same as fiat_shamir_preamble)
        let max_input_size = self.program_io.memory_layout.max_input_size;
        let max_output_size = self.program_io.memory_layout.max_output_size;
        let heap_size = self.program_io.memory_layout.heap_size;
        let inputs_hex = bytes_to_hex(&self.program_io.inputs);
        let outputs_hex = bytes_to_hex(&self.program_io.outputs);
        let panic = if self.program_io.panic { 1u64 } else { 0u64 };

        // Serialize commitment bytes for export
        let commitment_bytes: Vec<String> = self
            .proof
            .commitments
            .iter()
            .map(|c| {
                let mut buf = vec![];
                c.serialize_uncompressed(&mut buf).unwrap();
                buf.reverse(); // LE to BE for EVM
                bytes_to_hex(&buf)
            })
            .collect();

        // Run full verification with flush recording
        fiat_shamir_preamble(
            &self.program_io,
            self.proof.ram_K,
            self.proof.trace_length,
            self.preprocessing.shared.bytecode.entry_address,
            &mut self.transcript,
        );

        for commitment in &self.proof.commitments {
            self.transcript
                .append_serializable(b"commitment", commitment);
        }
        if let Some(ref untrusted_advice_commitment) = self.proof.untrusted_advice_commitment {
            self.transcript
                .append_serializable(b"untrusted_advice", untrusted_advice_commitment);
        }
        if let Some(ref trusted_advice_commitment) = self.trusted_advice_commitment {
            self.transcript
                .append_serializable(b"trusted_advice", trusted_advice_commitment);
        }

        // Capture transcript state after preamble + all commitments
        let capture_state =
            |t: &ProofTranscript, states: &mut Vec<String>, rounds: &mut Vec<u32>| {
                states.push(bytes_to_hex(t.state_bytes()));
                rounds.push(t.n_rounds_value());
            };
        capture_state(
            &self.transcript,
            &mut transcript_states,
            &mut transcript_n_rounds,
        );

        let (stage1_result, uniskip_challenge1) = self.verify_stage1()?;
        capture_state(
            &self.transcript,
            &mut transcript_states,
            &mut transcript_n_rounds,
        );

        let (stage2_result, uniskip_challenge2) = self.verify_stage2()?;
        capture_state(
            &self.transcript,
            &mut transcript_states,
            &mut transcript_n_rounds,
        );

        let stage3_result = self.verify_stage3()?;
        capture_state(
            &self.transcript,
            &mut transcript_states,
            &mut transcript_n_rounds,
        );

        let stage4_result = self.verify_stage4()?;
        capture_state(
            &self.transcript,
            &mut transcript_states,
            &mut transcript_n_rounds,
        );

        let stage5_result = self.verify_stage5()?;
        capture_state(
            &self.transcript,
            &mut transcript_states,
            &mut transcript_n_rounds,
        );

        let stage6_result = self.verify_stage6()?;
        capture_state(
            &self.transcript,
            &mut transcript_states,
            &mut transcript_n_rounds,
        );

        let stage7_result = self.verify_stage7()?;
        capture_state(
            &self.transcript,
            &mut transcript_states,
            &mut transcript_n_rounds,
        );

        // Clone transcript before stage 8 for Dory witness extraction
        let transcript_before_stage8 = self.transcript.clone();

        let stage8_data = self.verify_stage8()?;
        capture_state(
            &self.transcript,
            &mut transcript_states,
            &mut transcript_n_rounds,
        );

        let challenge_to_hex = |c: &F::Challenge| -> String {
            let f: F = (*c).into();
            fr_to_hex(&f)
        };

        let flush_history = self
            .opening_accumulator
            .flush_history
            .as_ref()
            .unwrap()
            .iter()
            .map(|batch| batch.iter().map(|c| fr_to_hex(c)).collect())
            .collect();

        let sumcheck_input_claims = self
            .opening_accumulator
            .sumcheck_input_claims_history
            .as_ref()
            .unwrap()
            .iter()
            .map(|batch| batch.iter().map(|c| fr_to_hex(c)).collect())
            .collect();

        use crate::zkvm::onchain_export::{
            OneHotConfigExport, RwConfigExport, StageInstanceConfig,
        };

        let rw = &self.proof.rw_config;
        let ohp = &self.one_hot_params;
        let log_T = trace_length.log_2();

        let rw_config_export = RwConfigExport {
            ram_rw_phase1_num_rounds: rw.ram_rw_phase1_num_rounds,
            ram_rw_phase2_num_rounds: rw.ram_rw_phase2_num_rounds,
            registers_rw_phase1_num_rounds: rw.registers_rw_phase1_num_rounds,
            registers_rw_phase2_num_rounds: rw.registers_rw_phase2_num_rounds,
        };

        let one_hot_config_export = OneHotConfigExport {
            log_k_chunk: ohp.log_k_chunk,
            lookups_ra_virtual_log_k_chunk: ohp.lookups_ra_virtual_log_k_chunk,
            k_chunk: ohp.k_chunk,
            ram_k: ohp.ram_k,
            bytecode_k: ohp.bytecode_k,
            instruction_d: ohp.instruction_d,
            bytecode_d: ohp.bytecode_d,
            ram_d: ohp.ram_d,
        };

        // Compute per-stage instance configs
        let log_K_ram = ram_k.log_2();
        let log_K_registers = common::constants::REGISTER_COUNT.ilog2() as usize;
        // InstructionReadRaf uses LOG_K = XLEN * 2 (from instruction_lookups/mod.rs)
        let log_K_instr = common::constants::XLEN * 2;

        let n_cycle_vars = log_T;

        // Stage 1: 1 instance (OuterRemaining)
        let stage1_config = StageInstanceConfig {
            num_rounds: vec![1 + n_cycle_vars],
            max_degree: 3,
        };

        // Stage 2: 5 instances (RamRW, ProductVirtRemainder, InstrClaimReduction, RamRafEval, OutputSumcheck)
        let ram_rw_total = log_K_ram + n_cycle_vars;
        let product_virt_rounds = n_cycle_vars;
        let instr_cr_rounds = n_cycle_vars;
        let phase1 = self.proof.rw_config.ram_rw_phase1_num_rounds as usize;
        let ram_raf_rounds = log_K_ram + n_cycle_vars - phase1;
        let output_rounds = log_K_ram + n_cycle_vars - phase1;
        let stage2_config = StageInstanceConfig {
            num_rounds: vec![
                ram_rw_total,
                product_virt_rounds,
                instr_cr_rounds,
                ram_raf_rounds,
                output_rounds,
            ],
            max_degree: 3,
        };

        // Stage 3: 3 instances (Shift, InstructionInput, RegistersClaimReduction)
        let stage3_config = StageInstanceConfig {
            num_rounds: vec![n_cycle_vars, n_cycle_vars, n_cycle_vars],
            max_degree: 3,
        };

        // Stage 4: 2 instances (RegistersRW, RamValCheck)
        let reg_rw_total = log_K_registers + n_cycle_vars;
        let ram_val_rounds = n_cycle_vars;
        let stage4_config = StageInstanceConfig {
            num_rounds: vec![reg_rw_total, ram_val_rounds],
            max_degree: 3,
        };

        // Stage 5: 3 instances (InstructionReadRaf, RamRaReduction, RegistersValEval)
        let instr_raf_rounds = log_K_instr + n_cycle_vars;
        let ram_ra_rounds = n_cycle_vars;
        let reg_val_rounds = n_cycle_vars;
        let n_virtual_ra_polys = log_K_instr / ohp.lookups_ra_virtual_log_k_chunk;
        let stage5_config = StageInstanceConfig {
            num_rounds: vec![instr_raf_rounds, ram_ra_rounds, reg_val_rounds],
            max_degree: n_virtual_ra_polys + 2, // eq * product(RA_chunks) * (val + gamma*raf)
        };

        // Stage 6: 6+ instances (BytecodeReadRaf, Booleanity, HammingBooleanity, RamRaVirtual, LookupsRaVirtual, IncReduction + optional advice)
        let bytecode_raf_rounds = ohp.bytecode_k.log_2() + n_cycle_vars;
        let booleanity_rounds = ohp.log_k_chunk + n_cycle_vars;
        let hamming_bool_rounds = n_cycle_vars;
        let ram_ra_virt_rounds = n_cycle_vars; // RamRaVirtual only uses log_T rounds
        let lookups_ra_virt_rounds = n_cycle_vars; // LookupsRaVirtual only uses log_T rounds
        let inc_rounds = n_cycle_vars;
        let mut stage6_num_rounds = vec![
            bytecode_raf_rounds,
            booleanity_rounds,
            hamming_bool_rounds,
            ram_ra_virt_rounds,
            lookups_ra_virt_rounds,
            inc_rounds,
        ];
        if self.trusted_advice_commitment.is_some() {
            stage6_num_rounds.push(n_cycle_vars);
        }
        if self.proof.untrusted_advice_commitment.is_some() {
            stage6_num_rounds.push(n_cycle_vars);
        }
        let stage6_max_degree = *[
            ohp.bytecode_d + 1,                             // BytecodeReadRaf
            ohp.instruction_d + ohp.bytecode_d + ohp.ram_d, // Booleanity: totalD
            3, // HammingBooleanity, RamRaVirtual, LookupsRaVirtual, IncReduction
        ]
        .iter()
        .max()
        .unwrap();
        let stage6_config = StageInstanceConfig {
            num_rounds: stage6_num_rounds,
            max_degree: stage6_max_degree,
        };

        // Stage 7: 1+ instances (HammingWeightClaimReduction + optional advice address phase)
        let hamming_weight_rounds = ohp.log_k_chunk;
        let mut stage7_num_rounds = vec![hamming_weight_rounds];
        if self.trusted_advice_commitment.is_some() {
            stage7_num_rounds.push(n_cycle_vars);
        }
        if self.proof.untrusted_advice_commitment.is_some() {
            stage7_num_rounds.push(n_cycle_vars);
        }
        let stage7_config = StageInstanceConfig {
            num_rounds: stage7_num_rounds,
            max_degree: 2,
        };

        let stage_instance_configs = vec![
            stage1_config,
            stage2_config,
            stage3_config,
            stage4_config,
            stage5_config,
            stage6_config,
            stage7_config,
        ];

        // Per-stage intermediate values: virtual polynomial evaluations + reference points
        use crate::poly::opening_proof::OpeningAccumulator;
        use crate::poly::opening_proof::SumcheckId;
        use crate::zkvm::witness::VirtualPolynomial;

        let point_to_hex = |point: &crate::poly::opening_proof::OpeningPoint<
            { crate::poly::opening_proof::BIG_ENDIAN },
            F,
        >|
         -> Vec<String> {
            point
                .r
                .iter()
                .map(|c| {
                    let f: F = (*c).into();
                    fr_to_hex(&f)
                })
                .collect()
        };

        // Stage 2 intermediate values: preprocessing evaluations for RafEval and OutputSumcheck
        let mut stage2_intermediates: Vec<(String, String)> = Vec::new();
        {
            use crate::poly::identity_poly::UnmapRamAddressPolynomial;
            use crate::poly::multilinear_polynomial::PolynomialEvaluation;
            use crate::poly::opening_proof::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN};
            use crate::poly::range_mask_polynomial::RangeMaskPolynomial;
            use crate::zkvm::ram::{eval_io_mle, remap_address};
            use common::constants::RAM_START_ADDRESS;

            let rw = &self.proof.rw_config;
            let phase1 = rw.ram_rw_phase1_num_rounds as usize;
            let phase2 = rw.ram_rw_phase2_num_rounds as usize;
            let log_K = log_K_ram;
            let phase3_cycle_rounds = log_T - phase1;

            // RafEvaluation and OutputSumcheck share the same challenge region in Stage 2.
            // Both have round_offset = max_num_rounds - (log_T + log_K) + phase1.
            let max_stage2_rounds = ram_rw_total; // RamRW has the most rounds in Stage 2
            let stage2_total = log_T + log_K;
            let offset = max_stage2_rounds - stage2_total + phase1;
            let raf_num_rounds = log_T + log_K - phase1;
            let r_slice = &stage2_result.challenges[offset..offset + raf_num_rounds];

            // normalize_opening_point: extract address challenges, skip cycle gap
            let addr_challenges = [
                r_slice[..phase2].to_vec(),
                r_slice[phase2 + phase3_cycle_rounds..].to_vec(),
            ]
            .concat();
            let r_address_be: OpeningPoint<BIG_ENDIAN, F> =
                OpeningPoint::<LITTLE_ENDIAN, F>::new(addr_challenges).match_endianness();

            // unmapEval
            let start_address = self.program_io.memory_layout.get_lowest_address();
            let unmap_eval =
                UnmapRamAddressPolynomial::<F>::new(log_K, start_address).evaluate(&r_address_be.r);
            stage2_intermediates.push(("unmapEval".to_string(), fr_to_hex(&unmap_eval)));

            // ioMaskEval and valIoEval use the same normalized address point
            let input_start_remapped = remap_address(
                self.program_io.memory_layout.input_start,
                &self.program_io.memory_layout,
            )
            .unwrap() as u128;
            let ram_start_remapped =
                remap_address(RAM_START_ADDRESS, &self.program_io.memory_layout).unwrap() as u128;
            let io_mask = RangeMaskPolynomial::<F>::new(input_start_remapped, ram_start_remapped);
            let io_mask_eval = io_mask.evaluate_mle(&r_address_be.r);
            stage2_intermediates.push(("ioMaskEval".to_string(), fr_to_hex(&io_mask_eval)));

            let val_io_eval: F = eval_io_mle::<F>(&self.program_io, &r_address_be.r);
            stage2_intermediates.push(("valIoEval".to_string(), fr_to_hex(&val_io_eval)));

            // Debug: export per-instance expected output claims and flush count
            if let Some(ref history) = self.opening_accumulator.flush_history {
                if history.len() > 3 {
                    let stage2_flush = &history[3];
                    stage2_intermediates.push((
                        "flushCount".to_string(),
                        format!("0x{:064x}", stage2_flush.len()),
                    ));
                }
            }
            // Per-instance expected output claims from Stage 2 (index 1 in per_instance_output_claims)
            if let Some(ref pic) = self.opening_accumulator.per_instance_output_claims {
                // pic[0] = Stage 1, pic[1] = Stage 2
                if pic.len() > 1 {
                    for (i, claim) in pic[1].iter().enumerate() {
                        stage2_intermediates
                            .push((format!("expectedOutput_{i}"), fr_to_hex(claim)));
                    }
                }
            }
            // Export individual accumulated opening claim values for flush order verification
            let get_virt_val = |vp: VirtualPolynomial, sc: SumcheckId| -> F {
                self.opening_accumulator
                    .get_virtual_polynomial_opening(vp, sc)
                    .1
            };
            use crate::zkvm::witness::CommittedPolynomial as CP;
            let get_comm_val = |cp: CP, sc: SumcheckId| -> F {
                self.opening_accumulator
                    .get_committed_polynomial_opening(cp, sc)
                    .1
            };
            use crate::zkvm::instruction::{CircuitFlags, InstructionFlags};
            // Instance 0 (RamRW) claims
            stage2_intermediates.push((
                "fc_RamVal".to_string(),
                fr_to_hex(&get_virt_val(
                    VirtualPolynomial::RamVal,
                    SumcheckId::RamReadWriteChecking,
                )),
            ));
            stage2_intermediates.push((
                "fc_RamRa".to_string(),
                fr_to_hex(&get_virt_val(
                    VirtualPolynomial::RamRa,
                    SumcheckId::RamReadWriteChecking,
                )),
            ));
            stage2_intermediates.push((
                "fc_RamInc".to_string(),
                fr_to_hex(&get_comm_val(CP::RamInc, SumcheckId::RamReadWriteChecking)),
            ));
            // Instance 1 (ProductVirt) claims - PRODUCT_UNIQUE_FACTOR_VIRTUALS order
            stage2_intermediates.push((
                "fc_LeftInstInput".to_string(),
                fr_to_hex(&get_virt_val(
                    VirtualPolynomial::LeftInstructionInput,
                    SumcheckId::SpartanProductVirtualization,
                )),
            ));
            stage2_intermediates.push((
                "fc_RightInstInput".to_string(),
                fr_to_hex(&get_virt_val(
                    VirtualPolynomial::RightInstructionInput,
                    SumcheckId::SpartanProductVirtualization,
                )),
            ));
            stage2_intermediates.push((
                "fc_Jump".to_string(),
                fr_to_hex(&get_virt_val(
                    VirtualPolynomial::OpFlags(CircuitFlags::Jump),
                    SumcheckId::SpartanProductVirtualization,
                )),
            ));
            stage2_intermediates.push((
                "fc_WriteLookupToRD".to_string(),
                fr_to_hex(&get_virt_val(
                    VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
                    SumcheckId::SpartanProductVirtualization,
                )),
            ));
            stage2_intermediates.push((
                "fc_LookupOutput".to_string(),
                fr_to_hex(&get_virt_val(
                    VirtualPolynomial::LookupOutput,
                    SumcheckId::SpartanProductVirtualization,
                )),
            ));
            stage2_intermediates.push((
                "fc_Branch".to_string(),
                fr_to_hex(&get_virt_val(
                    VirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
                    SumcheckId::SpartanProductVirtualization,
                )),
            ));
            stage2_intermediates.push((
                "fc_NextIsNoop".to_string(),
                fr_to_hex(&get_virt_val(
                    VirtualPolynomial::NextIsNoop,
                    SumcheckId::SpartanProductVirtualization,
                )),
            ));
            stage2_intermediates.push((
                "fc_VirtualInst".to_string(),
                fr_to_hex(&get_virt_val(
                    VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
                    SumcheckId::SpartanProductVirtualization,
                )),
            ));
            // Instance 2 (InstrCR) claims - only non-deduped ones
            stage2_intermediates.push((
                "fc_LeftLookupOperand".to_string(),
                fr_to_hex(&get_virt_val(
                    VirtualPolynomial::LeftLookupOperand,
                    SumcheckId::InstructionClaimReduction,
                )),
            ));
            stage2_intermediates.push((
                "fc_RightLookupOperand".to_string(),
                fr_to_hex(&get_virt_val(
                    VirtualPolynomial::RightLookupOperand,
                    SumcheckId::InstructionClaimReduction,
                )),
            ));
            // Instance 3 (RafEval)
            stage2_intermediates.push((
                "fc_RamRaRaf".to_string(),
                fr_to_hex(&get_virt_val(
                    VirtualPolynomial::RamRa,
                    SumcheckId::RamRafEvaluation,
                )),
            ));
            // Instance 4 (OutputSumcheck)
            stage2_intermediates.push((
                "fc_RamValFinal".to_string(),
                fr_to_hex(&get_virt_val(
                    VirtualPolynomial::RamValFinal,
                    SumcheckId::RamOutputCheck,
                )),
            ));

            // ProductVirt debug: export tauHigh, r0, intermediate factors
            {
                use crate::poly::lagrange_poly::LagrangePolynomial;
                let product_r0 = self
                    .opening_accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::UnivariateSkip,
                        SumcheckId::SpartanProductVirtualization,
                    )
                    .0;
                let r0_f: F = product_r0.r[0].into();
                stage2_intermediates.push(("prodVirt_r0".to_string(), fr_to_hex(&r0_f)));

                let r_cycle = self
                    .opening_accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::Product,
                        SumcheckId::SpartanOuter,
                    )
                    .0;
                // tau = r_cycle ++ [tau_high], tau_high was challenge_scalar_optimized
                // We can compute tau_high from tauHigh = transcript state, but it was sampled earlier.
                // Instead, export the r_cycle to verify it matches rCycleStage1
                for (i, c) in r_cycle.r.iter().enumerate() {
                    let f: F = (*c).into();
                    stage2_intermediates.push((format!("prodVirt_rCycle_{i}"), fr_to_hex(&f)));
                }

                // Compute intermediate factors for instance 1
                let l_inst = get_virt_val(
                    VirtualPolynomial::LeftInstructionInput,
                    SumcheckId::SpartanProductVirtualization,
                );
                let r_inst = get_virt_val(
                    VirtualPolynomial::RightInstructionInput,
                    SumcheckId::SpartanProductVirtualization,
                );
                let j_flag = get_virt_val(
                    VirtualPolynomial::OpFlags(CircuitFlags::Jump),
                    SumcheckId::SpartanProductVirtualization,
                );
                let lookup_out = get_virt_val(
                    VirtualPolynomial::LookupOutput,
                    SumcheckId::SpartanProductVirtualization,
                );
                let branch_flag = get_virt_val(
                    VirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
                    SumcheckId::SpartanProductVirtualization,
                );
                let next_is_noop = get_virt_val(
                    VirtualPolynomial::NextIsNoop,
                    SumcheckId::SpartanProductVirtualization,
                );

                let w = LagrangePolynomial::<F>::evals::<F::Challenge, 3>(&product_r0.r[0]);
                for (i, wi) in w.iter().enumerate() {
                    stage2_intermediates.push((format!("prodVirt_w{i}"), fr_to_hex(wi)));
                }

                let fused_left = w[0] * l_inst + w[1] * lookup_out + w[2] * j_flag;
                let fused_right =
                    w[0] * r_inst + w[1] * branch_flag + w[2] * (F::one() - next_is_noop);
                stage2_intermediates
                    .push(("prodVirt_fusedLeft".to_string(), fr_to_hex(&fused_left)));
                stage2_intermediates
                    .push(("prodVirt_fusedRight".to_string(), fr_to_hex(&fused_right)));
            }
        }

        // Stage 3 intermediate values: 10 virtual polynomial evaluations + 4 reference points
        let mut stage3_intermediates: Vec<(String, String)> = Vec::new();
        {
            let get_virt = |vp: VirtualPolynomial, sc: SumcheckId| -> (String, F) {
                let (_point, val) = self
                    .opening_accumulator
                    .get_virtual_polynomial_opening(vp, sc);
                let hex = fr_to_hex(&val);
                (hex, val)
            };
            let (hex, _) = get_virt(
                VirtualPolynomial::NextUnexpandedPC,
                SumcheckId::SpartanOuter,
            );
            stage3_intermediates.push(("nextUnexpandedPC".to_string(), hex));
            let (hex, _) = get_virt(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
            stage3_intermediates.push(("nextPC".to_string(), hex));
            let (hex, _) = get_virt(VirtualPolynomial::NextIsVirtual, SumcheckId::SpartanOuter);
            stage3_intermediates.push(("nextIsVirtual".to_string(), hex));
            let (hex, _) = get_virt(
                VirtualPolynomial::NextIsFirstInSequence,
                SumcheckId::SpartanOuter,
            );
            stage3_intermediates.push(("nextIsFirstInSeq".to_string(), hex));
            let (hex, _) = get_virt(
                VirtualPolynomial::NextIsNoop,
                SumcheckId::SpartanProductVirtualization,
            );
            stage3_intermediates.push(("nextIsNoop".to_string(), hex));
            let (hex, _) = get_virt(
                VirtualPolynomial::RightInstructionInput,
                SumcheckId::SpartanProductVirtualization,
            );
            stage3_intermediates.push(("rightInstructionInput".to_string(), hex));
            let (hex, _) = get_virt(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::SpartanProductVirtualization,
            );
            stage3_intermediates.push(("leftInstructionInput".to_string(), hex));
            let (hex, _) = get_virt(VirtualPolynomial::RdWriteValue, SumcheckId::SpartanOuter);
            stage3_intermediates.push(("rdWriteValue".to_string(), hex));
            let (hex, _) = get_virt(VirtualPolynomial::Rs1Value, SumcheckId::SpartanOuter);
            stage3_intermediates.push(("rs1Value".to_string(), hex));
            let (hex, _) = get_virt(VirtualPolynomial::Rs2Value, SumcheckId::SpartanOuter);
            stage3_intermediates.push(("rs2Value".to_string(), hex));

            // Reference points — serialize as comma-separated hex values
            let (point, _) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::NextPC,
                SumcheckId::SpartanOuter,
            );
            for (i, h) in point_to_hex(&point).into_iter().enumerate() {
                stage3_intermediates.push((format!("rOuter_{i}"), h));
            }
            let (point, _) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::SpartanProductVirtualization,
            );
            for (i, h) in point_to_hex(&point).into_iter().enumerate() {
                stage3_intermediates.push((format!("rProduct_{i}"), h));
            }
        }

        // Stage 4 intermediate values
        let mut stage4_intermediates: Vec<(String, String)> = Vec::new();
        {
            use crate::zkvm::ram;
            let log_K_ram = ram_k.log_2();

            // initEval: preprocessing-dependent initial RAM MLE evaluation
            let (ram_val_point, ram_val) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RamVal,
                SumcheckId::RamReadWriteChecking,
            );
            let r_address: Vec<F::Challenge> = ram_val_point.r[..log_K_ram].to_vec();
            let init_eval = ram::eval_initial_ram_mle::<F>(
                &self.preprocessing.shared.ram,
                &self.program_io,
                &r_address,
            );
            stage4_intermediates.push(("initEval".to_string(), fr_to_hex(&init_eval)));

            // Opening values needed for stage 4 input claims
            stage4_intermediates.push(("ramVal".to_string(), fr_to_hex(&ram_val)));
            let (_, ram_val_final) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            );
            stage4_intermediates.push(("ramValFinal".to_string(), fr_to_hex(&ram_val_final)));

            // rdWriteValue, rs1Value, rs2Value for RegistersRW input claim
            let (_, rdw) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RdWriteValue,
                SumcheckId::RegistersClaimReduction,
            );
            stage4_intermediates.push(("rdWriteValue".to_string(), fr_to_hex(&rdw)));
            let (_, rs1v) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::Rs1Value,
                SumcheckId::InstructionInputVirtualization,
            );
            stage4_intermediates.push(("rs1Value".to_string(), fr_to_hex(&rs1v)));
            let (_, rs2v) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::Rs2Value,
                SumcheckId::InstructionInputVirtualization,
            );
            stage4_intermediates.push(("rs2Value".to_string(), fr_to_hex(&rs2v)));

            // rCycleStage3: reversed stage 3 challenges (normalized opening point)
            for (i, c) in stage3_result.challenges.iter().rev().enumerate() {
                let f: F = (*c).into();
                stage4_intermediates.push((format!("rCycleStage3_{i}"), fr_to_hex(&f)));
            }

            // rCycleStage2Ram: cycle portion of RamRa @ RamReadWriteChecking point
            let (ram_ra_rw_point, _) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RamRa,
                SumcheckId::RamReadWriteChecking,
            );
            for (i, c) in ram_ra_rw_point.r[log_K_ram..].iter().enumerate() {
                let f: F = (*c).into();
                stage4_intermediates.push((format!("rCycleStage2Ram_{i}"), fr_to_hex(&f)));
            }
        }

        // Stage 5 intermediate values
        let mut stage5_intermediates: Vec<(String, String)> = Vec::new();
        {
            use crate::poly::identity_poly::{IdentityPolynomial, OperandPolynomial, OperandSide};
            use crate::poly::multilinear_polynomial::PolynomialEvaluation;
            use crate::zkvm::lookup_table::LookupTables;
            use strum::IntoEnumIterator;
            let log_K_instr = common::constants::XLEN * 2;
            let r_address_prime: Vec<F::Challenge> =
                stage5_result.challenges[..log_K_instr].to_vec();

            // Lookup table MLE evaluations
            for (i, table) in LookupTables::<{ common::constants::XLEN }>::iter().enumerate() {
                let val: F = table.evaluate_mle::<F, F::Challenge>(&r_address_prime);
                stage5_intermediates.push((format!("valEval_{i}"), fr_to_hex(&val)));
            }

            // Operand polynomial evaluations
            let left_eval: F = OperandPolynomial::<F>::new(log_K_instr, OperandSide::Left)
                .evaluate(&r_address_prime);
            stage5_intermediates.push(("leftOperandEval".to_string(), fr_to_hex(&left_eval)));
            let right_eval: F = OperandPolynomial::<F>::new(log_K_instr, OperandSide::Right)
                .evaluate(&r_address_prime);
            stage5_intermediates.push(("rightOperandEval".to_string(), fr_to_hex(&right_eval)));

            // Identity polynomial evaluation
            let id_eval: F = IdentityPolynomial::<F>::new(log_K_instr).evaluate(&r_address_prime);
            stage5_intermediates.push(("identityEval".to_string(), fr_to_hex(&id_eval)));

            // InstructionReadRaf input claim values
            let (_, lookup_output) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanProductVirtualization,
            );
            stage5_intermediates.push(("lookupOutput".to_string(), fr_to_hex(&lookup_output)));
            let (_, left_op) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::LeftLookupOperand,
                SumcheckId::InstructionClaimReduction,
            );
            stage5_intermediates.push(("leftOperand".to_string(), fr_to_hex(&left_op)));
            let (_, right_op) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RightLookupOperand,
                SumcheckId::InstructionClaimReduction,
            );
            stage5_intermediates.push(("rightOperand".to_string(), fr_to_hex(&right_op)));

            // rReduction: opening point for LookupOutput @ InstructionClaimReduction
            let (lo_point, _) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::LeftLookupOperand,
                SumcheckId::InstructionClaimReduction,
            );
            for (i, c) in lo_point.r.iter().enumerate() {
                let f: F = (*c).into();
                stage5_intermediates.push((format!("rReduction_{i}"), fr_to_hex(&f)));
            }

            // RamRa claim reduction input values
            let (ram_ra_raf_point, claim_raf) =
                self.opening_accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RamRa,
                    SumcheckId::RamRafEvaluation,
                );
            stage5_intermediates.push(("claimRaf".to_string(), fr_to_hex(&claim_raf)));
            let (ram_ra_rw_point, claim_rw) =
                self.opening_accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RamRa,
                    SumcheckId::RamReadWriteChecking,
                );
            stage5_intermediates.push(("claimRw".to_string(), fr_to_hex(&claim_rw)));
            let (ram_ra_val_point, claim_val) = self
                .opening_accumulator
                .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamValCheck);
            stage5_intermediates.push(("claimVal".to_string(), fr_to_hex(&claim_val)));

            // RamRa cycle points (last nCycleVars of each opening point)
            let log_K_ram = ram_k.log_2();
            for (i, c) in ram_ra_raf_point.r[log_K_ram..].iter().enumerate() {
                let f: F = (*c).into();
                stage5_intermediates.push((format!("rCycleRaf_{i}"), fr_to_hex(&f)));
            }
            for (i, c) in ram_ra_rw_point.r[log_K_ram..].iter().enumerate() {
                let f: F = (*c).into();
                stage5_intermediates.push((format!("rCycleRw_{i}"), fr_to_hex(&f)));
            }
            for (i, c) in ram_ra_val_point.r[log_K_ram..].iter().enumerate() {
                let f: F = (*c).into();
                stage5_intermediates.push((format!("rCycleVal_{i}"), fr_to_hex(&f)));
            }

            // RegistersValEvaluation input
            let (reg_val_point, registers_val) =
                self.opening_accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RegistersVal,
                    SumcheckId::RegistersReadWriteChecking,
                );
            stage5_intermediates.push(("registersVal".to_string(), fr_to_hex(&registers_val)));
            let log_K_registers = common::constants::REGISTER_COUNT.ilog2() as usize;
            for (i, c) in reg_val_point.r[log_K_registers..].iter().enumerate() {
                let f: F = (*c).into();
                stage5_intermediates.push((format!("rCycleStage4Reg_{i}"), fr_to_hex(&f)));
            }
        }

        // Stage 6 intermediate values
        let mut stage6_intermediates: Vec<(String, String)> = Vec::new();
        {
            let entry_idx = self.preprocessing.shared.bytecode.entry_bytecode_index();
            stage6_intermediates.push(("entryBytecodeIndex".to_string(), format!("{entry_idx}")));

            // BytecodeReadRaf: val_poly evaluations at normalized address point
            if let Some(val_evals) = &self.stage6_val_poly_evals {
                for (i, val) in val_evals.iter().enumerate() {
                    stage6_intermediates.push((format!("valPolyEval_{i}"), fr_to_hex(val)));
                }
            }

            // BytecodeReadRaf: rCycles (5 cycle reference points from prior stages)
            // These are the cycle portions of the virtual polynomial opening points
            // that bytecodeReadRaf references
            // rCycle1: Imm @ SpartanOuter
            let (rc1, _) = self
                .opening_accumulator
                .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
            for (i, c) in rc1.r.iter().enumerate() {
                let f: F = (*c).into();
                stage6_intermediates.push((format!("rCycle1_{i}"), fr_to_hex(&f)));
            }
            // rCycle2: OpFlags(Jump) @ SpartanProductVirtualization
            let (rc2, _) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(crate::zkvm::instruction::CircuitFlags::Jump),
                SumcheckId::SpartanProductVirtualization,
            );
            for (i, c) in rc2.r.iter().enumerate() {
                let f: F = (*c).into();
                stage6_intermediates.push((format!("rCycle2_{i}"), fr_to_hex(&f)));
            }
            // rCycle3: UnexpandedPC @ SpartanShift
            let (rc3, _) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::UnexpandedPC,
                SumcheckId::SpartanShift,
            );
            for (i, c) in rc3.r.iter().enumerate() {
                let f: F = (*c).into();
                stage6_intermediates.push((format!("rCycle3_{i}"), fr_to_hex(&f)));
            }
            // rCycle4: Rs1Ra @ RegistersReadWriteChecking (cycle = after register address split)
            let (rc4_full, _) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::Rs1Ra,
                SumcheckId::RegistersReadWriteChecking,
            );
            let reg_addr_len = (common::constants::REGISTER_COUNT as usize).log_2();
            for (i, c) in rc4_full.r[reg_addr_len..].iter().enumerate() {
                let f: F = (*c).into();
                stage6_intermediates.push((format!("rCycle4_{i}"), fr_to_hex(&f)));
            }
            // rCycle5: RdWa @ RegistersValEvaluation (cycle = after register address split)
            let (rc5_full, _) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RdWa,
                SumcheckId::RegistersValEvaluation,
            );
            for (i, c) in rc5_full.r[reg_addr_len..].iter().enumerate() {
                let f: F = (*c).into();
                stage6_intermediates.push((format!("rCycle5_{i}"), fr_to_hex(&f)));
            }

            // HammingBooleanity: rCycleHamming uses LookupOutput @ SpartanOuter
            // (matching expected_output_claim in hamming_booleanity.rs)
            let (hamming_r_cycle, _) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            );
            for (i, c) in hamming_r_cycle.r.iter().enumerate() {
                let f: F = (*c).into();
                stage6_intermediates.push((format!("rCycleHamming_{i}"), fr_to_hex(&f)));
            }

            // Booleanity + LookupsRaVirtual share source: InstructionRa(0) @ InstructionReadRaf
            let log_K_chunk = self.one_hot_params.log_k_chunk;
            let ra_virt_log_k = self.one_hot_params.lookups_ra_virtual_log_k_chunk;
            let (instr_ra0_point, _) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRa(0),
                SumcheckId::InstructionReadRaf,
            );

            // Booleanity: r_address from last log_k_chunk of LE address, r_cycle from cycle portion
            // Point is BE: [addr_BE(ra_virt_log_k) | cycle_BE(nCycleVars)]
            // The Booleanity expected_output_claim uses:
            //   eq(challenges, [r_address.reversed, r_cycle.reversed])
            // where r_address/r_cycle are stored LE internally.
            // For mleSliceReversed(ref, challenges, offset, len) = eq(ref, challenges_slice_reversed):
            //   we need ref in BE (= LE reversed) to get eq(BE, challenges_reversed_to_BE) = eq(LE_rev, LE_rev) = eq(LE, LE)
            let mut stage5_addr_le: Vec<F::Challenge> = instr_ra0_point.r[..ra_virt_log_k].to_vec();
            stage5_addr_le.reverse(); // BE -> LE
            let r_addr_bool_le = &stage5_addr_le[stage5_addr_le.len() - log_K_chunk..];
            // Export as BE for mleSliceReversed
            for (i, c) in r_addr_bool_le.iter().rev().enumerate() {
                let f: F = (*c).into();
                stage6_intermediates.push((format!("rAddressBool_{i}"), fr_to_hex(&f)));
            }
            // r_cycle: stored as BE in the OpeningPoint, export directly as BE
            for (i, c) in instr_ra0_point.r[ra_virt_log_k..].iter().enumerate() {
                let f: F = (*c).into();
                stage6_intermediates.push((format!("rCycleBool_{i}"), fr_to_hex(&f)));
            }

            // RamRaVirtual: ramRaInputClaim + rCycleRamRa
            let (ram_ra_cr_point, ram_ra_input_claim) =
                self.opening_accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RamRa,
                    SumcheckId::RamRaClaimReduction,
                );
            stage6_intermediates.push((
                "ramRaInputClaim".to_string(),
                fr_to_hex(&ram_ra_input_claim),
            ));
            let log_K_ram = ram_k.log_2();
            for (i, c) in ram_ra_cr_point.r[log_K_ram..].iter().enumerate() {
                let f: F = (*c).into();
                stage6_intermediates.push((format!("rCycleRamRa_{i}"), fr_to_hex(&f)));
            }

            // LookupsRaVirtual: cycle from same InstructionRa(0) @ InstructionReadRaf point
            let (_, lookups_r_cycle) = instr_ra0_point.split_at(ra_virt_log_k);
            for (i, c) in lookups_r_cycle.r.iter().enumerate() {
                let f: F = (*c).into();
                stage6_intermediates.push((format!("rCycleLookupsRa_{i}"), fr_to_hex(&f)));
            }

            // IncClaimReduction values
            let (_, inc_v1) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::RamReadWriteChecking,
            );
            stage6_intermediates.push(("incV1".to_string(), fr_to_hex(&inc_v1)));
            let (_, inc_v2) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::RamValCheck,
            );
            stage6_intermediates.push(("incV2".to_string(), fr_to_hex(&inc_v2)));
            let (_, inc_w1) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RdInc,
                SumcheckId::RegistersReadWriteChecking,
            );
            stage6_intermediates.push(("incW1".to_string(), fr_to_hex(&inc_w1)));
            let (_, inc_w2) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RdInc,
                SumcheckId::RegistersValEvaluation,
            );
            stage6_intermediates.push(("incW2".to_string(), fr_to_hex(&inc_w2)));

            // Inc reduction cycle points
            let (ram_inc_rw_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::RamReadWriteChecking,
            );
            // RamInc point is (log_K_ram + nCycleVars) in RamRW, cycle = last nCycleVars
            for (i, c) in ram_inc_rw_point.r[ram_inc_rw_point.r.len() - n_cycle_vars..]
                .iter()
                .enumerate()
            {
                let f: F = (*c).into();
                stage6_intermediates.push((format!("rCycleIncStage2_{i}"), fr_to_hex(&f)));
            }
            let (ram_inc_val_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::RamValCheck,
            );
            for (i, c) in ram_inc_val_point.r.iter().enumerate() {
                let f: F = (*c).into();
                stage6_intermediates.push((format!("rCycleIncStage4_{i}"), fr_to_hex(&f)));
            }
            let (rd_inc_rw_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RdInc,
                SumcheckId::RegistersReadWriteChecking,
            );
            for (i, c) in rd_inc_rw_point.r[rd_inc_rw_point.r.len() - n_cycle_vars..]
                .iter()
                .enumerate()
            {
                let f: F = (*c).into();
                stage6_intermediates.push((format!("sCycleIncStage4_{i}"), fr_to_hex(&f)));
            }
            let (rd_inc_val_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RdInc,
                SumcheckId::RegistersValEvaluation,
            );
            for (i, c) in rd_inc_val_point.r.iter().enumerate() {
                let f: F = (*c).into();
                stage6_intermediates.push((format!("sCycleIncStage5_{i}"), fr_to_hex(&f)));
            }
        }

        // Stage 7 intermediate values
        let mut stage7_intermediates: Vec<(String, String)> = Vec::new();
        {
            let n_instr = self.one_hot_params.instruction_d;
            let n_bytecode = self.one_hot_params.bytecode_d;
            let n_ram = self.one_hot_params.ram_d;
            let log_K_chunk = self.one_hot_params.log_k_chunk;

            // hwClaims: 1 for InstructionRa/BytecodeRa, ram_hw_factor for RamRa
            for i in 0..(n_instr + n_bytecode) {
                stage7_intermediates.push((format!("hwClaim_{i}"), fr_to_hex(&F::one())));
            }
            let (_, ram_hw_factor) = self.opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RamHammingWeight,
                SumcheckId::RamHammingBooleanity,
            );
            for i in 0..n_ram {
                stage7_intermediates.push((
                    format!("hwClaim_{}", n_instr + n_bytecode + i),
                    fr_to_hex(&ram_hw_factor),
                ));
            }

            // boolClaims: from Booleanity sumcheck flushed values
            // These are the committed Ra polynomial openings at the Booleanity point
            for i in 0..n_instr {
                let (_, v) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(i),
                    SumcheckId::Booleanity,
                );
                stage7_intermediates.push((format!("boolClaim_{i}"), fr_to_hex(&v)));
            }
            for i in 0..n_bytecode {
                let (_, v) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeRa(i),
                    SumcheckId::Booleanity,
                );
                stage7_intermediates.push((format!("boolClaim_{}", n_instr + i), fr_to_hex(&v)));
            }
            for i in 0..n_ram {
                let (_, v) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::RamRa(i),
                    SumcheckId::Booleanity,
                );
                stage7_intermediates.push((
                    format!("boolClaim_{}", n_instr + n_bytecode + i),
                    fr_to_hex(&v),
                ));
            }

            // virtClaims: from InstructionRaVirtualization/RamRaVirtualization
            for i in 0..n_instr {
                let (_, v) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(i),
                    SumcheckId::InstructionRaVirtualization,
                );
                stage7_intermediates.push((format!("virtClaim_{i}"), fr_to_hex(&v)));
            }
            for i in 0..n_bytecode {
                let (_, v) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeRa(i),
                    SumcheckId::BytecodeReadRaf,
                );
                stage7_intermediates.push((format!("virtClaim_{}", n_instr + i), fr_to_hex(&v)));
            }
            for i in 0..n_ram {
                let (_, v) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::RamRa(i),
                    SumcheckId::RamRaVirtualization,
                );
                stage7_intermediates.push((
                    format!("virtClaim_{}", n_instr + n_bytecode + i),
                    fr_to_hex(&v),
                ));
            }

            // rAddrBool: address portion of Booleanity point (first log_K_chunk)
            let (bool_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(0),
                SumcheckId::Booleanity,
            );
            for (i, c) in bool_point.r[..log_K_chunk].iter().enumerate() {
                let f: F = (*c).into();
                stage7_intermediates.push((format!("rAddrBool_{i}"), fr_to_hex(&f)));
            }

            // rAddrVirt: per-polynomial virtualization address points
            for i in 0..n_instr {
                let (point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(i),
                    SumcheckId::InstructionRaVirtualization,
                );
                for (j, c) in point.r[..log_K_chunk].iter().enumerate() {
                    let f: F = (*c).into();
                    stage7_intermediates.push((format!("rAddrVirt_{i}_{j}"), fr_to_hex(&f)));
                }
            }
            for i in 0..n_bytecode {
                let (point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeRa(i),
                    SumcheckId::BytecodeReadRaf,
                );
                for (j, c) in point.r[..log_K_chunk].iter().enumerate() {
                    let f: F = (*c).into();
                    stage7_intermediates
                        .push((format!("rAddrVirt_{}_{j}", n_instr + i), fr_to_hex(&f)));
                }
            }
            for i in 0..n_ram {
                let (point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::RamRa(i),
                    SumcheckId::RamRaVirtualization,
                );
                for (j, c) in point.r[..log_K_chunk].iter().enumerate() {
                    let f: F = (*c).into();
                    stage7_intermediates.push((
                        format!("rAddrVirt_{}_{j}", n_instr + n_bytecode + i),
                        fr_to_hex(&f),
                    ));
                }
            }
        }

        let stage_intermediate_values = vec![
            vec![],               // Stage 1
            stage2_intermediates, // Stage 2
            stage3_intermediates, // Stage 3
            stage4_intermediates, // Stage 4
            stage5_intermediates, // Stage 5
            stage6_intermediates, // Stage 6
            stage7_intermediates, // Stage 7
        ];

        // Extract Dory witness and Karabina commitment for on-chain MiMC InputHash
        let dory_export: (Option<serde_json::Value>, Vec<String>) = {
            use crate::poly::commitment::dory::extract_dory_witness;
            use crate::zkvm::onchain_export::dory_witness_to_json;
            use ark_bn254::Fq;
            use ark_ff::PrimeField;

            let mut replay_transcript = transcript_before_stage8;

            replay_transcript.append_scalars(b"rlc_claims", &stage8_data.claims);
            let _gamma_powers: Vec<F> =
                replay_transcript.challenge_scalar_powers(stage8_data.claims.len());

            // Reconstruct joint commitment from gamma_powers and commitments
            let joint_commitment = {
                let expected_polynomials = all_committed_polynomials(&self.one_hot_params);
                let mut commitments_map: HashMap<CommittedPolynomial, PCS::Commitment> =
                    HashMap::new();
                for (polynomial, commitment) in expected_polynomials
                    .into_iter()
                    .zip(&self.proof.commitments)
                {
                    commitments_map.insert(polynomial, commitment.clone());
                }
                if let Some(ref c) = self.trusted_advice_commitment {
                    commitments_map.insert(CommittedPolynomial::TrustedAdvice, c.clone());
                }
                if let Some(ref c) = self.proof.untrusted_advice_commitment {
                    commitments_map.insert(CommittedPolynomial::UntrustedAdvice, c.clone());
                }
                let mut rlc_map: HashMap<CommittedPolynomial, F> = HashMap::new();
                for (gamma, (poly, _)) in stage8_data
                    .gamma_powers
                    .iter()
                    .zip(stage8_data.polynomial_claims.iter())
                {
                    *rlc_map.entry(*poly).or_insert(F::zero()) += *gamma;
                }
                let (coeffs, coms): (Vec<F>, Vec<_>) = rlc_map
                    .into_iter()
                    .map(|(k, v)| (v, commitments_map.remove(&k).unwrap()))
                    .unzip();
                PCS::combine_commitments(&coms, &coeffs)
            };

            // SAFETY: These casts are sound because PCS = DoryCommitmentScheme
            // and verify_for_export is only called with Dory in non-ZK mode.
            let dory_proof: &crate::poly::commitment::dory::ArkDoryProof = unsafe {
                &*(&self.proof.joint_opening_proof as *const PCS::Proof
                    as *const crate::poly::commitment::dory::ArkDoryProof)
            };
            let dory_setup: &crate::poly::commitment::dory::ArkworksVerifierSetup = unsafe {
                &*(&self.preprocessing.generators as *const PCS::VerifierSetup
                    as *const crate::poly::commitment::dory::ArkworksVerifierSetup)
            };
            let dory_commitment: &crate::poly::commitment::dory::ArkGT = unsafe {
                &*(&joint_commitment as *const PCS::Commitment
                    as *const crate::poly::commitment::dory::ArkGT)
            };
            let dory_opening_point: &[<ark_bn254::Fr as JoltField>::Challenge] = unsafe {
                std::slice::from_raw_parts(
                    stage8_data.opening_point_challenges.as_ptr() as *const _,
                    stage8_data.opening_point_challenges.len(),
                )
            };
            let dory_joint_claim: &ark_bn254::Fr =
                unsafe { &*(&stage8_data.joint_claim as *const F as *const ark_bn254::Fr) };

            // Karabina-transformed commitment for MiMC InputHash
            let fq_hex = |f: &Fq| -> String {
                let bigint = f.into_bigint();
                let limbs = bigint.as_ref();
                let mut hex = String::new();
                for &limb in limbs.iter().rev() {
                    hex.push_str(&format!("{:016x}", limb));
                }
                format!("0x{}", hex)
            };
            let nine = Fq::from(9u64);
            let karabina = |a0: &Fq, a1: &Fq| -> String { fq_hex(&(*a0 - nine * a1)) };
            let c = &dory_commitment.0;
            let karabina_values = vec![
                karabina(&c.c0.c0.c0, &c.c0.c0.c1), // A0
                karabina(&c.c1.c0.c0, &c.c1.c0.c1), // A1
                karabina(&c.c0.c1.c0, &c.c0.c1.c1), // A2
                karabina(&c.c1.c1.c0, &c.c1.c1.c1), // A3
                karabina(&c.c0.c2.c0, &c.c0.c2.c1), // A4
                karabina(&c.c1.c2.c0, &c.c1.c2.c1), // A5
                fq_hex(&c.c0.c0.c1),                // A6
                fq_hex(&c.c1.c0.c1),                // A7
                fq_hex(&c.c0.c1.c1),                // A8
                fq_hex(&c.c1.c1.c1),                // A9
                fq_hex(&c.c0.c2.c1),                // A10
                fq_hex(&c.c1.c2.c1),                // A11
            ];

            let witness_json = match extract_dory_witness(
                dory_proof,
                dory_setup,
                &mut replay_transcript,
                dory_opening_point,
                dory_joint_claim,
                dory_commitment,
            ) {
                Ok(witness) => Some(dory_witness_to_json(&witness, dory_setup)),
                Err(e) => {
                    tracing::warn!("Failed to extract Dory witness: {:?}", e);
                    None
                }
            };

            (witness_json, karabina_values)
        };

        Ok(OnChainExportData {
            stage_challenges: vec![
                stage1_result
                    .challenges
                    .iter()
                    .map(challenge_to_hex)
                    .collect(),
                stage2_result
                    .challenges
                    .iter()
                    .map(challenge_to_hex)
                    .collect(),
                stage3_result
                    .challenges
                    .iter()
                    .map(challenge_to_hex)
                    .collect(),
                stage4_result
                    .challenges
                    .iter()
                    .map(challenge_to_hex)
                    .collect(),
                stage5_result
                    .challenges
                    .iter()
                    .map(challenge_to_hex)
                    .collect(),
                stage6_result
                    .challenges
                    .iter()
                    .map(challenge_to_hex)
                    .collect(),
                stage7_result
                    .challenges
                    .iter()
                    .map(challenge_to_hex)
                    .collect(),
            ],
            uniskip_polys: vec![uniskip1_coeffs, uniskip2_coeffs],
            uniskip_challenges: vec![
                challenge_to_hex(&uniskip_challenge1),
                challenge_to_hex(&uniskip_challenge2),
            ],
            stage_compressed_polys,
            sumcheck_input_claims,
            flush_history,
            opening_claims: self
                .proof
                .opening_claims
                .0
                .iter()
                .map(|(id, (_point, value))| (format!("{id:?}"), fr_to_hex(value)))
                .collect(),
            commitment_bytes,
            transcript_states,
            transcript_n_rounds,
            trace_length,
            ram_k,
            entry_address,
            max_input_size,
            max_output_size,
            heap_size,
            inputs_hex,
            outputs_hex,
            panic,
            num_rows_bits: self.spartan_key.num_rows_bits(),
            stage_instance_configs,
            stage_intermediate_values,
            rw_config: rw_config_export,
            one_hot_config: one_hot_config_export,
            accumulator_openings: self
                .opening_accumulator
                .openings
                .iter()
                .map(|(id, (point, value))| {
                    let point_hex: Vec<String> = point
                        .r
                        .iter()
                        .map(|c| {
                            let f: F = (*c).into();
                            fr_to_hex(&f)
                        })
                        .collect();
                    (format!("{id:?}"), point_hex, fr_to_hex(value))
                })
                .collect(),
            stage8_committed_claims: stage8_data.claims.iter().map(|c| fr_to_hex(c)).collect(),
            stage8_scaling_factors: stage8_data
                .scaling_factors
                .iter()
                .map(|c| fr_to_hex(c))
                .collect(),
            stage8_opening_point: stage8_data
                .opening_point
                .iter()
                .map(|c| fr_to_hex(c))
                .collect(),
            stage8_joint_claim: fr_to_hex(&stage8_data.joint_claim),
            dory_witness_json: dory_export.0,
            commitment_karabina: dory_export.1,
            combined_claims: self
                .opening_accumulator
                .combined_claims_history
                .as_ref()
                .unwrap()
                .iter()
                .map(|c| fr_to_hex(c))
                .collect(),
        })
    }

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn verify_stage1(&mut self) -> Result<(StageVerifyResult<F>, F::Challenge), ProofVerifyError> {
        let (uni_skip_params, uni_skip_challenge) = verify_stage1_uni_skip(
            &self.proof.stage1_uni_skip_first_round_proof,
            &self.spartan_key,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )?;

        // Drain uniskip OC block IDs (pending_claims were drained inside verify_transcript)
        #[cfg(feature = "zk")]
        let uniskip_oc_ids = self.opening_accumulator.take_pending_claim_ids();

        let spartan_outer_remaining = OuterRemainingSumcheckVerifier::new(
            self.spartan_key,
            self.proof.trace_length,
            &uni_skip_params,
            &self.opening_accumulator,
        );

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> =
            vec![&spartan_outer_remaining];

        let (batching_coefficients, r_stage1) = BatchedSumcheck::verify(
            &self.proof.stage1_sumcheck_proof,
            instances.clone(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        )?;

        #[cfg(feature = "zk")]
        {
            let regular_oc_ids = self.opening_accumulator.take_pending_claim_ids();

            let batched_output_constraint = batch_output_constraints(&instances);
            let batched_input_constraint = batch_input_constraints(&instances);

            let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();

            let output_constraint_challenge_values = if batched_output_constraint.is_some() {
                let mut values = batching_coefficients.clone();
                for instance in &instances {
                    let num_rounds = instance.num_rounds();
                    let offset = instance.round_offset(max_num_rounds);
                    let r_slice = &r_stage1[offset..offset + num_rounds];
                    values.extend(
                        instance
                            .get_params()
                            .output_constraint_challenge_values(r_slice),
                    );
                }
                values
            } else {
                Vec::new()
            };

            let mut input_constraint_challenge_values: Vec<F> =
                scale_batching_coefficients(&batching_coefficients, &instances);
            for instance in &instances {
                input_constraint_challenge_values.extend(
                    instance
                        .get_params()
                        .input_constraint_challenge_values(&self.opening_accumulator),
                );
            }

            let uniskip_input_constraint = uni_skip_params.input_claim_constraint();
            let uniskip_input_constraint_challenge_values =
                uni_skip_params.input_constraint_challenge_values(&self.opening_accumulator);

            let stage_result = StageVerifyResult::with_uniskip(
                r_stage1,
                batched_output_constraint,
                output_constraint_challenge_values,
                batched_input_constraint,
                input_constraint_challenge_values,
                uniskip_input_constraint,
                uniskip_input_constraint_challenge_values,
                vec![uniskip_oc_ids, regular_oc_ids],
            );

            Ok((stage_result, uni_skip_challenge))
        }
        #[cfg(not(feature = "zk"))]
        Ok((
            StageVerifyResult {
                challenges: r_stage1,
            },
            uni_skip_challenge,
        ))
    }

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn verify_stage2(&mut self) -> Result<(StageVerifyResult<F>, F::Challenge), ProofVerifyError> {
        let (uni_skip_params, uni_skip_challenge) = verify_stage2_uni_skip(
            &self.proof.stage2_uni_skip_first_round_proof,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )?;

        #[cfg(feature = "zk")]
        let uniskip_oc_ids = self.opening_accumulator.take_pending_claim_ids();

        let ram_read_write_checking = RamReadWriteCheckingVerifier::new(
            &self.opening_accumulator,
            &mut self.transcript,
            &self.one_hot_params,
            self.proof.trace_length,
            &self.proof.rw_config,
        );

        let spartan_product_virtual_remainder = ProductVirtualRemainderVerifier::new(
            self.proof.trace_length,
            uni_skip_params.clone(),
            &self.opening_accumulator,
        );

        let instruction_claim_reduction = InstructionLookupsClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let ram_raf_evaluation = RamRafEvaluationSumcheckVerifier::new(
            &self.program_io.memory_layout,
            &self.one_hot_params,
            self.proof.trace_length,
            &self.proof.rw_config,
            &self.opening_accumulator,
        );

        let ram_output_check = OutputSumcheckVerifier::new(
            self.proof.ram_K,
            &self.program_io,
            &mut self.transcript,
            self.proof.trace_length,
            &self.proof.rw_config,
        );

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> = vec![
            &ram_read_write_checking,
            &spartan_product_virtual_remainder,
            &instruction_claim_reduction,
            &ram_raf_evaluation,
            &ram_output_check,
        ];

        let (batching_coefficients, r_stage2) = BatchedSumcheck::verify(
            &self.proof.stage2_sumcheck_proof,
            instances.clone(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        )?;

        #[cfg(feature = "zk")]
        {
            let regular_oc_ids = self.opening_accumulator.take_pending_claim_ids();

            let batched_output_constraint = batch_output_constraints(&instances);
            let batched_input_constraint = batch_input_constraints(&instances);

            let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
            let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
            let mut input_constraint_challenge_values: Vec<F> =
                scale_batching_coefficients(&batching_coefficients, &instances);
            for instance in &instances {
                let num_rounds = instance.num_rounds();
                let offset = instance.round_offset(max_num_rounds);
                let r_slice = &r_stage2[offset..offset + num_rounds];
                output_constraint_challenge_values.extend(
                    instance
                        .get_params()
                        .output_constraint_challenge_values(r_slice),
                );
                input_constraint_challenge_values.extend(
                    instance
                        .get_params()
                        .input_constraint_challenge_values(&self.opening_accumulator),
                );
            }

            let uniskip_input_constraint = uni_skip_params.input_claim_constraint();
            let uniskip_input_constraint_challenge_values =
                uni_skip_params.input_constraint_challenge_values(&self.opening_accumulator);

            let stage_result = StageVerifyResult::with_uniskip(
                r_stage2,
                batched_output_constraint,
                output_constraint_challenge_values,
                batched_input_constraint,
                input_constraint_challenge_values,
                uniskip_input_constraint,
                uniskip_input_constraint_challenge_values,
                vec![uniskip_oc_ids, regular_oc_ids],
            );

            Ok((stage_result, uni_skip_challenge))
        }
        #[cfg(not(feature = "zk"))]
        Ok((
            StageVerifyResult {
                challenges: r_stage2,
            },
            uni_skip_challenge,
        ))
    }

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn verify_stage3(&mut self) -> Result<StageVerifyResult<F>, ProofVerifyError> {
        let spartan_shift = ShiftSumcheckVerifier::new(
            self.proof.trace_length.log_2(),
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let spartan_instruction_input =
            InstructionInputSumcheckVerifier::new(&self.opening_accumulator, &mut self.transcript);
        let spartan_registers_claim_reduction = RegistersClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> = vec![
            &spartan_shift,
            &spartan_instruction_input,
            &spartan_registers_claim_reduction,
        ];

        let (batching_coefficients, r_stage3) = BatchedSumcheck::verify(
            &self.proof.stage3_sumcheck_proof,
            instances.clone(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        )?;

        #[cfg(feature = "zk")]
        {
            let regular_oc_ids = self.opening_accumulator.take_pending_claim_ids();
            let batched_output_constraint = batch_output_constraints(&instances);
            let batched_input_constraint = batch_input_constraints(&instances);
            let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
            let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
            let mut input_constraint_challenge_values: Vec<F> =
                scale_batching_coefficients(&batching_coefficients, &instances);
            for instance in &instances {
                let num_rounds = instance.num_rounds();
                let offset = instance.round_offset(max_num_rounds);
                let r_slice = &r_stage3[offset..offset + num_rounds];
                output_constraint_challenge_values.extend(
                    instance
                        .get_params()
                        .output_constraint_challenge_values(r_slice),
                );
                input_constraint_challenge_values.extend(
                    instance
                        .get_params()
                        .input_constraint_challenge_values(&self.opening_accumulator),
                );
            }
            Ok(StageVerifyResult::new(
                r_stage3,
                batched_output_constraint,
                output_constraint_challenge_values,
                batched_input_constraint,
                input_constraint_challenge_values,
                vec![regular_oc_ids],
            ))
        }
        #[cfg(not(feature = "zk"))]
        Ok(StageVerifyResult {
            challenges: r_stage3,
        })
    }

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn verify_stage4(&mut self) -> Result<StageVerifyResult<F>, ProofVerifyError> {
        let registers_read_write_checking = RegistersReadWriteCheckingVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
            &self.proof.rw_config,
        );
        verifier_accumulate_advice::<F>(
            self.proof.ram_K,
            &self.program_io,
            self.proof.untrusted_advice_commitment.is_some(),
            self.trusted_advice_commitment.is_some(),
            &mut self.opening_accumulator,
        );
        // Domain-separate the batching challenge.
        self.transcript.append_bytes(b"ram_val_check_gamma", &[]);
        let ram_val_check_gamma: F = self.transcript.challenge_scalar::<F>();
        let initial_ram_state = crate::zkvm::ram::gen_ram_initial_memory_state::<F>(
            self.proof.ram_K,
            &self.preprocessing.shared.ram,
            &self.program_io,
        );
        let ram_val_check = RamValCheckSumcheckVerifier::new(
            &initial_ram_state,
            &self.program_io,
            &self.preprocessing.shared.ram,
            self.proof.trace_length,
            self.proof.ram_K,
            &self.proof.rw_config,
            ram_val_check_gamma,
            &self.opening_accumulator,
        );

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> =
            vec![&registers_read_write_checking, &ram_val_check];

        let (batching_coefficients, r_stage4) = BatchedSumcheck::verify(
            &self.proof.stage4_sumcheck_proof,
            instances.clone(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        )?;

        #[cfg(feature = "zk")]
        {
            let regular_oc_ids = self.opening_accumulator.take_pending_claim_ids();
            let batched_output_constraint = batch_output_constraints(&instances);
            let batched_input_constraint = batch_input_constraints(&instances);
            let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
            let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
            let mut input_constraint_challenge_values: Vec<F> =
                scale_batching_coefficients(&batching_coefficients, &instances);
            for instance in &instances {
                let num_rounds = instance.num_rounds();
                let offset = instance.round_offset(max_num_rounds);
                let r_slice = &r_stage4[offset..offset + num_rounds];
                output_constraint_challenge_values.extend(
                    instance
                        .get_params()
                        .output_constraint_challenge_values(r_slice),
                );
                input_constraint_challenge_values.extend(
                    instance
                        .get_params()
                        .input_constraint_challenge_values(&self.opening_accumulator),
                );
            }
            Ok(StageVerifyResult::new(
                r_stage4,
                batched_output_constraint,
                output_constraint_challenge_values,
                batched_input_constraint,
                input_constraint_challenge_values,
                vec![regular_oc_ids],
            ))
        }
        #[cfg(not(feature = "zk"))]
        Ok(StageVerifyResult {
            challenges: r_stage4,
        })
    }

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn verify_stage5(&mut self) -> Result<StageVerifyResult<F>, ProofVerifyError> {
        let n_cycle_vars = self.proof.trace_length.log_2();

        let lookups_read_raf = InstructionReadRafSumcheckVerifier::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let ram_ra_reduction = RamRaClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let registers_val_evaluation =
            RegistersValEvaluationSumcheckVerifier::new(&self.opening_accumulator);

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> = vec![
            &lookups_read_raf,
            &ram_ra_reduction,
            &registers_val_evaluation,
        ];

        let (batching_coefficients, r_stage5) = BatchedSumcheck::verify(
            &self.proof.stage5_sumcheck_proof,
            instances.clone(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        )?;

        #[cfg(feature = "zk")]
        {
            let regular_oc_ids = self.opening_accumulator.take_pending_claim_ids();
            let batched_output_constraint = batch_output_constraints(&instances);
            let batched_input_constraint = batch_input_constraints(&instances);
            let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
            let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
            let mut input_constraint_challenge_values: Vec<F> =
                scale_batching_coefficients(&batching_coefficients, &instances);
            for instance in &instances {
                let num_rounds = instance.num_rounds();
                let offset = instance.round_offset(max_num_rounds);
                let r_slice = &r_stage5[offset..offset + num_rounds];
                output_constraint_challenge_values.extend(
                    instance
                        .get_params()
                        .output_constraint_challenge_values(r_slice),
                );
                input_constraint_challenge_values.extend(
                    instance
                        .get_params()
                        .input_constraint_challenge_values(&self.opening_accumulator),
                );
            }
            Ok(StageVerifyResult::new(
                r_stage5,
                batched_output_constraint,
                output_constraint_challenge_values,
                batched_input_constraint,
                input_constraint_challenge_values,
                vec![regular_oc_ids],
            ))
        }
        #[cfg(not(feature = "zk"))]
        Ok(StageVerifyResult {
            challenges: r_stage5,
        })
    }

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn verify_stage6(&mut self) -> Result<StageVerifyResult<F>, ProofVerifyError> {
        let n_cycle_vars = self.proof.trace_length.log_2();
        let bytecode_read_raf = BytecodeReadRafSumcheckVerifier::gen(
            &self.preprocessing.shared.bytecode,
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let ram_hamming_booleanity =
            HammingBooleanitySumcheckVerifier::new(&self.opening_accumulator);
        let booleanity_params = BooleanitySumcheckParams::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let booleanity = BooleanitySumcheckVerifier::new(booleanity_params);
        let ram_ra_virtual = RamRaVirtualSumcheckVerifier::new(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_ra_virtual = LookupsRaSumcheckVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let inc_reduction = IncClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // Advice claim reduction (Phase 1 in Stage 6): trusted and untrusted are separate instances.
        if self.trusted_advice_commitment.is_some() {
            self.advice_reduction_verifier_trusted = Some(AdviceClaimReductionVerifier::new(
                AdviceKind::Trusted,
                &self.program_io.memory_layout,
                self.proof.trace_length,
                &self.opening_accumulator,
            ));
        }
        if self.proof.untrusted_advice_commitment.is_some() {
            self.advice_reduction_verifier_untrusted = Some(AdviceClaimReductionVerifier::new(
                AdviceKind::Untrusted,
                &self.program_io.memory_layout,
                self.proof.trace_length,
                &self.opening_accumulator,
            ));
        }

        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> = vec![
            &bytecode_read_raf,
            &booleanity,
            &ram_hamming_booleanity,
            &ram_ra_virtual,
            &lookups_ra_virtual,
            &inc_reduction,
        ];
        if let Some(ref advice) = self.advice_reduction_verifier_trusted {
            instances.push(advice);
        }
        if let Some(ref advice) = self.advice_reduction_verifier_untrusted {
            instances.push(advice);
        }

        let (batching_coefficients, r_stage6) = BatchedSumcheck::verify(
            &self.proof.stage6_sumcheck_proof,
            instances.clone(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        )?;

        #[cfg(feature = "zk")]
        {
            let regular_oc_ids = self.opening_accumulator.take_pending_claim_ids();
            let batched_output_constraint = batch_output_constraints(&instances);
            let batched_input_constraint = batch_input_constraints(&instances);
            let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
            let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
            let mut input_constraint_challenge_values: Vec<F> =
                scale_batching_coefficients(&batching_coefficients, &instances);
            for instance in &instances {
                let num_rounds = instance.num_rounds();
                let offset = instance.round_offset(max_num_rounds);
                let r_slice = &r_stage6[offset..offset + num_rounds];
                output_constraint_challenge_values.extend(
                    instance
                        .get_params()
                        .output_constraint_challenge_values(r_slice),
                );
                input_constraint_challenge_values.extend(
                    instance
                        .get_params()
                        .input_constraint_challenge_values(&self.opening_accumulator),
                );
            }
            Ok(StageVerifyResult::new(
                r_stage6,
                batched_output_constraint,
                output_constraint_challenge_values,
                batched_input_constraint,
                input_constraint_challenge_values,
                vec![regular_oc_ids],
            ))
        }
        #[cfg(not(feature = "zk"))]
        {
            use crate::poly::multilinear_polynomial::PolynomialEvaluation;
            use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
            let opening_point = bytecode_read_raf.params.normalize_opening_point(&r_stage6);
            let (r_address_prime, _) = opening_point.split_at(bytecode_read_raf.params.log_K);
            self.stage6_val_poly_evals = Some(std::array::from_fn(|i| {
                bytecode_read_raf.params.val_polys[i].evaluate(&r_address_prime.r)
            }));

            Ok(StageVerifyResult {
                challenges: r_stage6,
            })
        }
    }

    #[cfg(feature = "zk")]
    #[allow(clippy::too_many_arguments)]
    fn verify_blindfold(
        &mut self,
        sumcheck_challenges: &[Vec<F::Challenge>; 7],
        uniskip_challenges: [F::Challenge; 2],
        stage_output_constraints: &[Option<OutputClaimConstraint>; 7],
        output_constraint_challenge_values: &[Vec<F>; 7],
        stage_input_constraints: &[InputClaimConstraint; 7],
        input_constraint_challenge_values: &[Vec<F>; 7],
        // For stages 0-1: batched input constraint for regular rounds (different from uni-skip)
        stage1_batched_input: &InputClaimConstraint,
        stage2_batched_input: &InputClaimConstraint,
        stage1_batched_input_values: &[F],
        stage2_batched_input_values: &[F],
        stage8_data: &Stage8VerifyData<F>,
        oc_blocks: Vec<Vec<OpeningId>>,
    ) -> Result<(), ProofVerifyError> {
        // Build stage configurations including uni-skip rounds.
        // Uni-skip rounds are the first round of stages 1 and 2 (indices 0 and 1).
        let stage_proofs = [
            &self.proof.stage1_sumcheck_proof,
            &self.proof.stage2_sumcheck_proof,
            &self.proof.stage3_sumcheck_proof,
            &self.proof.stage4_sumcheck_proof,
            &self.proof.stage5_sumcheck_proof,
            &self.proof.stage6_sumcheck_proof,
            &self.proof.stage7_sumcheck_proof,
        ];

        // Precompute power sums for uni-skip domains
        let outer_power_sums = LagrangeHelper::power_sums::<
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
            OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
        >();
        let product_power_sums = LagrangeHelper::power_sums::<
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
            PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
        >();

        let mut stage_configs = Vec::new();
        // Track which stage_config index corresponds to uni-skip and regular first rounds
        let mut uniskip_indices: Vec<usize> = Vec::new(); // Only 2 elements for stages 0-1
        let mut regular_first_round_indices: Vec<usize> = Vec::new(); // 7 elements for all stages
        let mut last_round_indices: Vec<usize> = Vec::new();

        for (stage_idx, proof) in stage_proofs.iter().enumerate() {
            // For stages 0 and 1 (Jolt stages 1 and 2), add uni-skip config first
            if stage_idx < 2 {
                let uniskip_proof = if stage_idx == 0 {
                    &self.proof.stage1_uni_skip_first_round_proof
                } else {
                    &self.proof.stage2_uni_skip_first_round_proof
                };
                let poly_degree = uniskip_proof.poly_degree();

                let power_sums: Vec<i128> = if stage_idx == 0 {
                    outer_power_sums.to_vec()
                } else {
                    product_power_sums.to_vec()
                };

                // Record uni-skip index for its input constraint
                uniskip_indices.push(stage_configs.len());

                let config = if stage_idx == 0 {
                    StageConfig::new_uniskip(poly_degree, power_sums)
                } else {
                    StageConfig::new_uniskip_chain(poly_degree, power_sums)
                };
                stage_configs.push(config);
            }

            // Record first regular round index for its input constraint
            regular_first_round_indices.push(stage_configs.len());

            // Add regular sumcheck rounds
            let num_rounds = proof.num_rounds();
            for round_idx in 0..num_rounds {
                let poly_degree = match proof {
                    crate::subprotocols::sumcheck::SumcheckInstanceProof::Clear(std_proof) => {
                        std_proof.compressed_polys[round_idx]
                            .coeffs_except_linear_term
                            .len()
                    }
                    crate::subprotocols::sumcheck::SumcheckInstanceProof::Zk(zk_proof) => {
                        zk_proof.poly_degrees[round_idx]
                    }
                };
                // First regular round ALWAYS starts a new chain
                // (batched claims differ from uni-skip output due to batching coefficients)
                let starts_new_chain = round_idx == 0;
                let config = if starts_new_chain {
                    StageConfig::new_chain(1, poly_degree)
                } else {
                    StageConfig::new(1, poly_degree)
                };
                stage_configs.push(config);
            }

            // Record the last round index for output constraint
            last_round_indices.push(stage_configs.len() - 1);
        }

        // Add final_output configurations using the batched constraints from verifier instances
        for (stage_idx, constraint) in stage_output_constraints.iter().enumerate() {
            if let Some(batched) = constraint {
                let last_round_idx = last_round_indices[stage_idx];
                stage_configs[last_round_idx].final_output =
                    Some(ClaimBindingConfig::with_constraint(batched.clone()));
            }
        }

        // Add initial_input configurations for uni-skip stages (stages 0-1)
        // These use the uni-skip's own input constraints
        let uniskip_constraints = [
            stage_input_constraints[0].clone(), // Stage 0 uni-skip
            stage_input_constraints[1].clone(), // Stage 1 uni-skip
        ];
        for (i, constraint) in uniskip_constraints.iter().enumerate() {
            let idx = uniskip_indices[i];
            stage_configs[idx].initial_input =
                Some(ClaimBindingConfig::with_constraint(constraint.clone()));
        }

        // Add initial_input configurations for regular first rounds (all 7 stages)
        // These use the batched input constraints from the stage results
        let regular_constraints = [
            stage1_batched_input.clone(),       // Stage 0 regular
            stage2_batched_input.clone(),       // Stage 1 regular
            stage_input_constraints[2].clone(), // Stage 2
            stage_input_constraints[3].clone(), // Stage 3
            stage_input_constraints[4].clone(), // Stage 4
            stage_input_constraints[5].clone(), // Stage 5
            stage_input_constraints[6].clone(), // Stage 6
        ];
        for (i, constraint) in regular_constraints.iter().enumerate() {
            let idx = regular_first_round_indices[i];
            stage_configs[idx].initial_input =
                Some(ClaimBindingConfig::with_constraint(constraint.clone()));
        }

        let extra_constraint_terms: Vec<(ValueSource, ValueSource)> = stage8_data
            .opening_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (ValueSource::challenge(i), ValueSource::opening(*id)))
            .collect();
        let extra_constraint = OutputClaimConstraint::linear(extra_constraint_terms);
        let extra_constraints = vec![extra_constraint];

        // Build baked public inputs from expected values
        let mut baked_challenges: Vec<F> = Vec::new();
        for (stage_idx, stage_challenges) in sumcheck_challenges.iter().enumerate() {
            if stage_idx < 2 {
                baked_challenges.push(uniskip_challenges[stage_idx].into());
            }
            for challenge in stage_challenges.iter() {
                baked_challenges.push((*challenge).into());
            }
        }

        let all_input_challenge_values: [&[F]; 9] = [
            &input_constraint_challenge_values[0],
            stage1_batched_input_values,
            &input_constraint_challenge_values[1],
            stage2_batched_input_values,
            &input_constraint_challenge_values[2],
            &input_constraint_challenge_values[3],
            &input_constraint_challenge_values[4],
            &input_constraint_challenge_values[5],
            &input_constraint_challenge_values[6],
        ];
        let mut baked_input_challenges: Vec<F> = Vec::new();
        for expected_values in all_input_challenge_values.iter() {
            baked_input_challenges.extend_from_slice(expected_values);
        }

        let mut baked_output_challenges: Vec<F> = Vec::new();
        for expected_values in output_constraint_challenge_values.iter() {
            baked_output_challenges.extend_from_slice(expected_values);
        }

        let baked = BakedPublicInputs {
            challenges: baked_challenges,
            initial_claims: Vec::new(),
            batching_coefficients: Vec::new(),
            output_constraint_challenges: baked_output_challenges,
            input_constraint_challenges: baked_input_challenges,
            extra_constraint_challenges: stage8_data.constraint_coeffs.clone(),
        };

        let mut round_commitments: Vec<C::G1> = Vec::new();
        let mut oc_row_commitments: Vec<C::G1> = Vec::new();
        for (stage_idx, proof) in stage_proofs.iter().enumerate() {
            if stage_idx < 2 {
                let uniskip_proof = if stage_idx == 0 {
                    &self.proof.stage1_uni_skip_first_round_proof
                } else {
                    &self.proof.stage2_uni_skip_first_round_proof
                };
                if let UniSkipFirstRoundProofVariant::Zk(zk_uniskip) = uniskip_proof {
                    round_commitments.push(zk_uniskip.commitment);
                    oc_row_commitments.extend_from_slice(&zk_uniskip.output_claims_commitments);
                }
            }
            if let SumcheckInstanceProof::Zk(zk_proof) = proof {
                round_commitments.extend(zk_proof.round_commitments.iter().cloned());
                oc_row_commitments.extend_from_slice(&zk_proof.output_claims_commitments);
            }
        }

        let builder = VerifierR1CSBuilder::new_with_extra(
            &stage_configs,
            &extra_constraints,
            &baked,
            oc_blocks,
        );
        let r1cs = builder.build();

        let eval_commitment = PCS::eval_commitment(&self.proof.joint_opening_proof)
            .ok_or(ProofVerifyError::InvalidOpeningProof)?;
        let eval_commitments = vec![eval_commitment];

        let verifier_input = BlindFoldVerifierInput {
            round_commitments,
            output_claims_row_commitments: oc_row_commitments,
            eval_commitments,
        };

        let pedersen_generator_count = pedersen_generator_count_for_r1cs(&r1cs);
        let pedersen_generators = self
            .preprocessing
            .pedersen_generators(pedersen_generator_count);
        let eval_commitment_gens =
            PCS::eval_commitment_gens_verifier(&self.preprocessing.generators);
        let verifier =
            BlindFoldVerifier::<_, _>::new(&pedersen_generators, &r1cs, eval_commitment_gens);
        let mut blindfold_transcript = ProofTranscript::new(b"BlindFold");

        verifier
            .verify(
                &self.proof.blindfold_proof,
                &verifier_input,
                &mut blindfold_transcript,
            )
            .map_err(|e| ProofVerifyError::BlindFoldError(format!("{e:?}")))?;

        tracing::debug!(
            "BlindFold verification passed: {} R1CS constraints",
            r1cs.num_constraints
        );

        Ok(())
    }

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn verify_stage7(&mut self) -> Result<StageVerifyResult<F>, ProofVerifyError> {
        // Create verifier for HammingWeightClaimReduction
        // (r_cycle and r_addr_bool are extracted from Booleanity opening internally)
        let hw_verifier = HammingWeightClaimReductionVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> =
            vec![&hw_verifier];
        if let Some(advice_reduction_verifier_trusted) =
            self.advice_reduction_verifier_trusted.as_mut()
        {
            let mut params = advice_reduction_verifier_trusted.params.borrow_mut();
            if params.num_address_phase_rounds() > 0 {
                // Transition phase
                params.phase = ReductionPhase::AddressVariables;
                instances.push(advice_reduction_verifier_trusted);
            }
        }
        if let Some(advice_reduction_verifier_untrusted) =
            self.advice_reduction_verifier_untrusted.as_mut()
        {
            let mut params = advice_reduction_verifier_untrusted.params.borrow_mut();
            if params.num_address_phase_rounds() > 0 {
                // Transition phase
                params.phase = ReductionPhase::AddressVariables;
                instances.push(advice_reduction_verifier_untrusted);
            }
        }

        let (batching_coefficients, r_stage7) = BatchedSumcheck::verify(
            &self.proof.stage7_sumcheck_proof,
            instances.clone(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        )?;

        #[cfg(feature = "zk")]
        {
            let regular_oc_ids = self.opening_accumulator.take_pending_claim_ids();
            let batched_output_constraint = batch_output_constraints(&instances);
            let batched_input_constraint = batch_input_constraints(&instances);
            let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
            let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
            let mut input_constraint_challenge_values: Vec<F> =
                scale_batching_coefficients(&batching_coefficients, &instances);
            for instance in &instances {
                let num_rounds = instance.num_rounds();
                let offset = instance.round_offset(max_num_rounds);
                let r_slice = &r_stage7[offset..offset + num_rounds];
                output_constraint_challenge_values.extend(
                    instance
                        .get_params()
                        .output_constraint_challenge_values(r_slice),
                );
                input_constraint_challenge_values.extend(
                    instance
                        .get_params()
                        .input_constraint_challenge_values(&self.opening_accumulator),
                );
            }
            Ok(StageVerifyResult::new(
                r_stage7,
                batched_output_constraint,
                output_constraint_challenge_values,
                batched_input_constraint,
                input_constraint_challenge_values,
                vec![regular_oc_ids],
            ))
        }
        #[cfg(not(feature = "zk"))]
        Ok(StageVerifyResult {
            challenges: r_stage7,
        })
    }

    /// Stage 8: Dory batch opening verification.
    fn verify_stage8(&mut self) -> Result<Stage8VerifyData<F>, ProofVerifyError> {
        // Initialize DoryGlobals with the layout from the proof
        // This ensures the verifier uses the same layout as the prover
        let _guard = DoryGlobals::initialize_context(
            1 << self.one_hot_params.log_k_chunk,
            self.proof.trace_length.next_power_of_two(),
            DoryContext::Main,
            Some(self.proof.dory_layout),
        );

        // Get the unified opening point from HammingWeightClaimReduction
        // This contains (r_address_stage7 || r_cycle_stage6) in big-endian
        let (opening_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::HammingWeightClaimReduction,
        );
        let log_k_chunk = self.one_hot_params.log_k_chunk;
        let r_address_stage7 = &opening_point.r[..log_k_chunk];

        // 1. Collect all (polynomial, claim) pairs
        let mut polynomial_claims = Vec::new();
        let mut scaling_factors = Vec::new();

        // Dense polynomials: RamInc and RdInc (from IncClaimReduction in Stage 6)
        let (_, ram_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::IncClaimReduction,
        );
        let (_, rd_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
        );

        // Dense polynomials are zero-padded in the Dory matrix, so their evaluation
        // includes a factor eq(r_addr, 0) = ∏(1 − r_addr_i).
        let lagrange_factor: F = EqPolynomial::zero_selector(r_address_stage7);
        polynomial_claims.push((CommittedPolynomial::RamInc, ram_inc_claim * lagrange_factor));
        scaling_factors.push(lagrange_factor);
        polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim * lagrange_factor));
        scaling_factors.push(lagrange_factor);

        // Sparse polynomials: all RA polys (from HammingWeightClaimReduction)
        for i in 0..self.one_hot_params.instruction_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::InstructionRa(i), claim));
            scaling_factors.push(F::one());
        }
        for i in 0..self.one_hot_params.bytecode_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::BytecodeRa(i), claim));
            scaling_factors.push(F::one());
        }
        for i in 0..self.one_hot_params.ram_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::RamRa(i), claim));
            scaling_factors.push(F::one());
        }

        // Advice polynomials: TrustedAdvice and UntrustedAdvice (from AdviceClaimReduction in Stage 6)
        // These are committed with smaller dimensions, so we apply Lagrange factors to embed
        // them in the top-left block of the main Dory matrix.
        let mut include_trusted_advice = false;
        let mut include_untrusted_advice = false;

        if let Some((advice_point, advice_claim)) = self
            .opening_accumulator
            .get_advice_opening(AdviceKind::Trusted, SumcheckId::AdviceClaimReduction)
        {
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
            polynomial_claims.push((
                CommittedPolynomial::TrustedAdvice,
                advice_claim * lagrange_factor,
            ));
            scaling_factors.push(lagrange_factor);
            include_trusted_advice = true;
        }

        if let Some((advice_point, advice_claim)) = self
            .opening_accumulator
            .get_advice_opening(AdviceKind::Untrusted, SumcheckId::AdviceClaimReduction)
        {
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
            polynomial_claims.push((
                CommittedPolynomial::UntrustedAdvice,
                advice_claim * lagrange_factor,
            ));
            scaling_factors.push(lagrange_factor);
            include_untrusted_advice = true;
        }

        // 2. Sample gamma and compute powers for RLC
        let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
        // In non-ZK mode, absorb claims before sampling gamma for Fiat-Shamir binding.
        // In ZK mode, claims are secret; binding comes from BlindFold constraints instead.
        #[cfg(not(feature = "zk"))]
        self.transcript.append_scalars(b"rlc_claims", &claims);
        let gamma_powers: Vec<F> = self
            .transcript
            .challenge_scalar_powers(polynomial_claims.len());
        let constraint_coeffs: Vec<F> = gamma_powers
            .iter()
            .zip(&scaling_factors)
            .map(|(gamma, scale)| *gamma * *scale)
            .collect();

        let opening_ids = stage8_opening_ids(
            &self.one_hot_params,
            include_trusted_advice,
            include_untrusted_advice,
        );
        let joint_claim: F = gamma_powers
            .iter()
            .zip(claims.iter())
            .map(|(gamma, claim)| *gamma * claim)
            .sum();

        // Save polynomial_claims for export before moving into state
        let polynomial_claims_export = polynomial_claims.clone();

        // Build state for computing joint commitment/claim
        let state = DoryOpeningState {
            opening_point: opening_point.r.clone(),
            gamma_powers: gamma_powers.clone(),
            polynomial_claims,
        };

        // Build commitments map
        let mut commitments_map = HashMap::new();
        let expected_polynomials = all_committed_polynomials(&self.one_hot_params);
        if expected_polynomials.len() != self.proof.commitments.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                expected_polynomials.len(),
                self.proof.commitments.len(),
            ));
        }
        for (polynomial, commitment) in expected_polynomials
            .into_iter()
            .zip(&self.proof.commitments)
        {
            commitments_map.insert(polynomial, commitment.clone());
        }

        // Add advice commitments if they're part of the batch
        if let Some(ref commitment) = self.trusted_advice_commitment {
            if state
                .polynomial_claims
                .iter()
                .any(|(p, _)| *p == CommittedPolynomial::TrustedAdvice)
            {
                commitments_map.insert(CommittedPolynomial::TrustedAdvice, commitment.clone());
            }
        }
        if let Some(ref commitment) = self.proof.untrusted_advice_commitment {
            if state
                .polynomial_claims
                .iter()
                .any(|(p, _)| *p == CommittedPolynomial::UntrustedAdvice)
            {
                commitments_map.insert(CommittedPolynomial::UntrustedAdvice, commitment.clone());
            }
        }

        let joint_commitment = self.compute_joint_commitment(&mut commitments_map, &state)?;

        let zk_mode = self.opening_accumulator.zk_mode;
        if zk_mode {
            PCS::verify(
                &self.proof.joint_opening_proof,
                &self.preprocessing.generators,
                &mut self.transcript,
                &opening_point.r,
                &F::zero(),
                &joint_commitment,
            )?;

            #[cfg(feature = "zk")]
            {
                let y_com: C::G1 = PCS::eval_commitment(&self.proof.joint_opening_proof)
                    .ok_or(ProofVerifyError::InvalidOpeningProof)?;
                bind_opening_inputs_zk::<F, C, _>(&mut self.transcript, &opening_point.r, &y_com);
            }
            #[cfg(not(feature = "zk"))]
            {
                return Err(ProofVerifyError::ZkFeatureRequired);
            }
        } else {
            PCS::verify(
                &self.proof.joint_opening_proof,
                &self.preprocessing.generators,
                &mut self.transcript,
                &opening_point.r,
                &joint_claim,
                &joint_commitment,
            )?;

            bind_opening_inputs::<F, _>(&mut self.transcript, &opening_point.r, &joint_claim);
        }

        let opening_point_scalars: Vec<F> = opening_point.r.iter().map(|c| (*c).into()).collect();

        Ok(Stage8VerifyData {
            opening_ids,
            constraint_coeffs,
            claims,
            scaling_factors,
            opening_point: opening_point_scalars,
            opening_point_challenges: opening_point.r.clone(),
            joint_claim,
            gamma_powers,
            polynomial_claims: polynomial_claims_export,
        })
    }

    /// Compute joint commitment for the batch opening.
    fn compute_joint_commitment(
        &self,
        commitment_map: &mut HashMap<CommittedPolynomial, PCS::Commitment>,
        state: &DoryOpeningState<F>,
    ) -> Result<PCS::Commitment, ProofVerifyError> {
        let mut rlc_map = HashMap::new();
        for (gamma, (poly, _claim)) in state
            .gamma_powers
            .iter()
            .zip(state.polynomial_claims.iter())
        {
            *rlc_map.entry(*poly).or_insert(F::zero()) += *gamma;
        }

        let (coeffs, commitments): (Vec<F>, Vec<PCS::Commitment>) = rlc_map
            .into_iter()
            .map(|(k, v)| {
                commitment_map
                    .remove(&k)
                    .map(|c| (v, c))
                    .ok_or(ProofVerifyError::InternalError)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();

        Ok(PCS::combine_commitments(&commitments, &coeffs))
    }
}

#[derive(Debug, Clone)]
pub struct JoltSharedPreprocessing {
    pub bytecode: Arc<BytecodePreprocessing>,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
    pub max_padded_trace_length: usize,
}

impl CanonicalSerialize for JoltSharedPreprocessing {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.bytecode
            .as_ref()
            .serialize_with_mode(&mut writer, compress)?;
        self.ram.serialize_with_mode(&mut writer, compress)?;
        self.memory_layout
            .serialize_with_mode(&mut writer, compress)?;
        self.max_padded_trace_length
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.bytecode.serialized_size(compress)
            + self.ram.serialized_size(compress)
            + self.memory_layout.serialized_size(compress)
            + self.max_padded_trace_length.serialized_size(compress)
    }
}

impl CanonicalDeserialize for JoltSharedPreprocessing {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let bytecode =
            BytecodePreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
        let ram = RAMPreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
        let memory_layout = MemoryLayout::deserialize_with_mode(&mut reader, compress, validate)?;
        let max_padded_trace_length =
            usize::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(Self {
            bytecode: Arc::new(bytecode),
            ram,
            memory_layout,
            max_padded_trace_length,
        })
    }
}

impl ark_serialize::Valid for JoltSharedPreprocessing {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.bytecode.check()?;
        self.ram.check()?;
        self.memory_layout.check()?;
        Ok(())
    }
}

impl JoltSharedPreprocessing {
    #[tracing::instrument(skip_all, name = "JoltSharedPreprocessing::new")]
    pub fn new(
        bytecode: Vec<Instruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_padded_trace_length: usize,
        entry_address: u64,
    ) -> Result<JoltSharedPreprocessing, PreprocessingError> {
        let bytecode = Arc::new(BytecodePreprocessing::preprocess(bytecode, entry_address)?);
        let ram = RAMPreprocessing::preprocess(memory_init);
        Ok(Self {
            bytecode,
            ram,
            memory_layout,
            max_padded_trace_length,
        })
    }
}

/// Serializable wrapper around [`PedersenGenerators`] for ZK setup transfer.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BlindfoldSetup<C: JoltCurve>(pub PedersenGenerators<C>);

impl<C: JoltCurve> std::ops::Deref for BlindfoldSetup<C> {
    type Target = PedersenGenerators<C>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<C: JoltCurve> From<BlindfoldSetup<C>> for PedersenGenerators<C> {
    fn from(setup: BlindfoldSetup<C>) -> Self {
        setup.0
    }
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub shared: JoltSharedPreprocessing,
    pub blindfold_setup: Option<BlindfoldSetup<C>>,
}

impl<F, C, PCS> Serializable for JoltVerifierPreprocessing<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
{
}

impl<F, C, PCS> JoltVerifierPreprocessing<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn save_to_target_dir(&self, target_dir: &str) -> std::io::Result<()> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::create(filename.as_path())?;
        let mut data = Vec::new();
        self.serialize_compressed(&mut data).unwrap();
        file.write_all(&data)?;
        Ok(())
    }

    pub fn read_from_target_dir(target_dir: &str) -> std::io::Result<Self> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::open(filename.as_path())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self::deserialize_compressed(&*data).unwrap())
    }
}

impl<F: JoltField, C: JoltCurve<F = F>, PCS: CommitmentScheme<Field = F>>
    JoltVerifierPreprocessing<F, C, PCS>
{
    #[tracing::instrument(skip_all, name = "JoltVerifierPreprocessing::new")]
    pub fn new(
        shared: JoltSharedPreprocessing,
        generators: PCS::VerifierSetup,
        blindfold_setup: Option<BlindfoldSetup<C>>,
    ) -> Self {
        Self {
            generators,
            shared,
            blindfold_setup,
        }
    }

    #[cfg(feature = "zk")]
    pub fn pedersen_generators(&self, count: usize) -> PedersenGenerators<C> {
        let gens = &self
            .blindfold_setup
            .as_ref()
            .expect("BlindfoldSetup required for ZK mode")
            .0;
        assert!(
            count <= gens.message_generators.len(),
            "Requested {count} Pedersen generators but BlindfoldSetup only has {}",
            gens.message_generators.len()
        );
        PedersenGenerators::new(
            gens.message_generators[..count].to_vec(),
            gens.blinding_generator,
        )
    }
}

#[cfg(feature = "prover")]
impl<F: JoltField, C: JoltCurve<F = F>, PCS: CommitmentScheme<Field = F> + ZkEvalCommitment<C>>
    From<&JoltProverPreprocessing<F, C, PCS>> for JoltVerifierPreprocessing<F, C, PCS>
{
    fn from(prover_preprocessing: &JoltProverPreprocessing<F, C, PCS>) -> Self {
        let shared = prover_preprocessing.shared.clone();
        let generators = PCS::setup_verifier(&prover_preprocessing.generators);
        #[cfg(not(feature = "zk"))]
        let blindfold_setup = None;
        #[cfg(feature = "zk")]
        let blindfold_setup = Some(prover_preprocessing.blindfold_setup());
        Self::new(shared, generators, blindfold_setup)
    }
}
