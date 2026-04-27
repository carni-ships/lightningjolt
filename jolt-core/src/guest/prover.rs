use super::program::Program;
use crate::curve::{Bn254Curve, JoltCurve};
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{
    BatchOpeningScheme, CommitmentScheme, StreamingCommitmentScheme, ZkEvalCommitment,
};
use crate::poly::commitment::dory::DoryCommitmentScheme;
use crate::transcripts::Transcript;
use crate::zkvm::bytecode::PreprocessingError;
use crate::zkvm::proof_serialization::JoltProof;
use crate::zkvm::prover::JoltProverPreprocessing;
use crate::zkvm::ProverDebugInfo;
use common::jolt_device::MemoryLayout;
use tracer::{JoltDevice, LazyTraceIterator};

#[allow(clippy::type_complexity)]
#[cfg(feature = "prover")]
pub fn preprocess(
    guest: &Program,
    max_trace_length: usize,
) -> Result<
    JoltProverPreprocessing<ark_bn254::Fr, Bn254Curve, DoryCommitmentScheme>,
    PreprocessingError,
> {
    use crate::zkvm::verifier::JoltSharedPreprocessing;

    let (bytecode, memory_init, program_size, e_entry) = guest.decode();

    let mut memory_config = guest.memory_config;
    memory_config.program_size = Some(program_size);
    let memory_layout = MemoryLayout::new(&memory_config);
    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode,
        memory_layout,
        memory_init,
        max_trace_length,
        e_entry,
    )?;
    Ok(JoltProverPreprocessing::new(shared_preprocessing))
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
#[cfg(feature = "prover")]
pub fn prove<
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: StreamingCommitmentScheme<Field = F> + ZkEvalCommitment<C> + BatchOpeningScheme<F>,
    FS: Transcript,
>(
    guest: &Program,
    inputs_bytes: &[u8],
    untrusted_advice_bytes: &[u8],
    trusted_advice_bytes: &[u8],
    trusted_advice_commitment: Option<<PCS as CommitmentScheme>::Commitment>,
    trusted_advice_hint: Option<<PCS as CommitmentScheme>::OpeningProofHint>,
    output_bytes: &mut [u8],
    preprocessing: &JoltProverPreprocessing<F, C, PCS>,
) -> (
    JoltProof<F, C, PCS, FS>,
    JoltDevice,
    Option<ProverDebugInfo<F, FS, PCS>>,
) {
    use crate::zkvm::prover::JoltCpuProver;

    let prover = JoltCpuProver::<F, C, PCS, FS>::gen_from_elf(
        preprocessing,
        &guest.elf_contents,
        inputs_bytes,
        untrusted_advice_bytes,
        trusted_advice_bytes,
        trusted_advice_commitment,
        trusted_advice_hint,
        None,
    );
    let io_device = prover.program_ios[0].clone();
    let (proof, debug_info) = prover.prove();
    output_bytes[..io_device.outputs.len()].copy_from_slice(&io_device.outputs);
    (proof, io_device, debug_info)
}

/// Batch proving for multiple transactions together through Stages 1-7,
/// then a single combined Stage 8 opening proof.
/// This enables true batch proving where all transactions are combined into
/// a single Stage 5 (sumcheck) and single Stage 8 (opening) proof.
///
/// # Arguments
/// * `lazy_traces` - Lazy trace iterators for each transaction (used for trace generation)
/// * `traces` - Pre-materialized traces for each transaction (already padded to same length)
/// * `program_ios` - JoltDevice for each transaction (with different I/O)
/// * `trusted_advice_commitments_batch` - Per-transaction advice commitments (can be None)
/// * `trusted_advice_hints_batch` - Per-transaction advice hints (can be None)
/// * `final_memory_states` - Final memory state for each transaction
/// * `gamma_powers` - Powers of gamma for RLC combination (should have len = num_txs)
/// * `output_bytes` - Output buffer (uses first transaction's output)
/// * `preprocessing` - Shared preprocessing
///
/// # Returns
/// Tuple of (combined_proof, first_io_device, debug_info)
#[allow(clippy::type_complexity)]
#[cfg(feature = "prover")]
pub fn prove_batch<F: JoltField, C: JoltCurve<F = F>, PCS, FS>(
    lazy_traces: Vec<LazyTraceIterator>,
    traces: Vec<Vec<tracer::instruction::Cycle>>,
    program_ios: Vec<JoltDevice>,
    trusted_advice_commitments_batch: Vec<Option<<PCS as CommitmentScheme>::Commitment>>,
    trusted_advice_hints_batch: Vec<Option<<PCS as CommitmentScheme>::OpeningProofHint>>,
    final_memory_states: Vec<tracer::emulator::memory::Memory>,
    gamma_powers: Vec<F>,
    output_bytes: &mut [u8],
    preprocessing: &JoltProverPreprocessing<F, C, PCS>,
) -> (
    JoltProof<F, C, PCS, FS>,
    JoltDevice,
    Option<ProverDebugInfo<F, FS, PCS>>,
)
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: StreamingCommitmentScheme<Field = F> + ZkEvalCommitment<C> + BatchOpeningScheme<F>,
    FS: Transcript,
{
    use tracer::instruction::Cycle;
    use crate::zkvm::prover::JoltCpuProver;

    assert!(!traces.is_empty(), "Must have at least one trace for batch proving");
    assert_eq!(traces.len(), program_ios.len(), "traces and program_ios must have same length");

    let prover = JoltCpuProver::<F, C, PCS, FS>::gen_from_traces_batch(
        preprocessing,
        lazy_traces,
        traces,
        program_ios,
        trusted_advice_commitments_batch,
        trusted_advice_hints_batch,
        final_memory_states,
        gamma_powers,
    );

    let io_device = prover.program_ios[0].clone();
    let (proof, debug_info) = prover.prove();
    output_bytes[..io_device.outputs.len()].copy_from_slice(&io_device.outputs);
    (proof, io_device, debug_info)
}
