#[cfg(not(feature = "zk"))]
use std::collections::BTreeMap;
use std::io::{Read, Write};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use num::FromPrimitive;
use strum::EnumCount;

#[cfg(not(feature = "zk"))]
use crate::poly::opening_proof::{OpeningPoint, Openings};
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::BlindFoldProof;
use crate::{
    curve::JoltCurve,
    field::JoltField,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, dory::DoryLayout},
        opening_proof::{OpeningId, PolynomialId, SumcheckId},
    },
};
use crate::{
    subprotocols::{
        sumcheck::SumcheckInstanceProof, univariate_skip::UniSkipFirstRoundProofVariant,
    },
    transcripts::Transcript,
    zkvm::{
        config::{OneHotConfig, ReadWriteConfig},
        instruction::{CircuitFlags, InstructionFlags},
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};

pub struct JoltProof<
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
> {
    pub commitments: Vec<PCS::Commitment>,
    pub stage1_uni_skip_first_round_proof: UniSkipFirstRoundProofVariant<F, C, FS>,
    pub stage1_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage2_uni_skip_first_round_proof: UniSkipFirstRoundProofVariant<F, C, FS>,
    pub stage2_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage3_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage4_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage5_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage6_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    pub stage7_sumcheck_proof: SumcheckInstanceProof<F, C, FS>,
    #[cfg(feature = "zk")]
    pub blindfold_proof: BlindFoldProof<F, C>,
    pub joint_opening_proof: PCS::Proof,
    /// Opening proof hint computed during Stage 8 Dory Opening.
    /// Stored for cross-transaction aggregation; not serialized (not needed for verification/transmission).
    pub opening_hint: Option<PCS::OpeningProofHint>,
    pub untrusted_advice_commitment: Option<PCS::Commitment>,
    #[cfg(not(feature = "zk"))]
    pub opening_claims: Claims<F>,
    pub trace_length: usize,
    pub ram_K: usize,
    pub rw_config: ReadWriteConfig,
    pub one_hot_config: OneHotConfig,
    pub dory_layout: DoryLayout,
}

impl<F: JoltField, C: JoltCurve<F = F>, PCS: CommitmentScheme<Field = F>, FS: Transcript>
    CanonicalSerialize for JoltProof<F, C, PCS, FS>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.commitments.serialize_with_mode(&mut writer, compress)?;
        self.stage1_uni_skip_first_round_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage1_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2_uni_skip_first_round_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage3_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage4_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage5_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage6_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage7_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        #[cfg(feature = "zk")]
        self.blindfold_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.joint_opening_proof
            .serialize_with_mode(&mut writer, compress)?;
        // NOTE: opening_hint is intentionally skipped -- it is for local
        // cross-transaction aggregation use only and is not needed for
        // verification or transmission.
        self.untrusted_advice_commitment
            .serialize_with_mode(&mut writer, compress)?;
        #[cfg(not(feature = "zk"))]
        self.opening_claims.serialize_with_mode(&mut writer, compress)?;
        (self.trace_length as u64).serialize_with_mode(&mut writer, compress)?;
        (self.ram_K as u64).serialize_with_mode(&mut writer, compress)?;
        self.rw_config.serialize_with_mode(&mut writer, compress)?;
        self.one_hot_config.serialize_with_mode(&mut writer, compress)?;
        self.dory_layout.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let mut size = self.commitments.serialized_size(compress);
        size += self
            .stage1_uni_skip_first_round_proof
            .serialized_size(compress);
        size += self.stage1_sumcheck_proof.serialized_size(compress);
        size += self
            .stage2_uni_skip_first_round_proof
            .serialized_size(compress);
        size += self.stage2_sumcheck_proof.serialized_size(compress);
        size += self.stage3_sumcheck_proof.serialized_size(compress);
        size += self.stage4_sumcheck_proof.serialized_size(compress);
        size += self.stage5_sumcheck_proof.serialized_size(compress);
        size += self.stage6_sumcheck_proof.serialized_size(compress);
        size += self.stage7_sumcheck_proof.serialized_size(compress);
        #[cfg(feature = "zk")]
        {
            size += self.blindfold_proof.serialized_size(compress);
        }
        size += self.joint_opening_proof.serialized_size(compress);
        // NOTE: opening_hint is skipped
        size += self
            .untrusted_advice_commitment
            .serialized_size(compress);
        #[cfg(not(feature = "zk"))]
        {
            size += self.opening_claims.serialized_size(compress);
        }
        size += (self.trace_length as u64).serialized_size(compress);
        size += (self.ram_K as u64).serialized_size(compress);
        size += self.rw_config.serialized_size(compress);
        size += self.one_hot_config.serialized_size(compress);
        size += self.dory_layout.serialized_size(compress);
        size
    }
}

impl<F: JoltField, C: JoltCurve<F = F>, PCS: CommitmentScheme<Field = F>, FS: Transcript>
    Valid for JoltProof<F, C, PCS, FS>
{
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: JoltField, C: JoltCurve<F = F>, PCS: CommitmentScheme<Field = F>, FS: Transcript>
    CanonicalDeserialize for JoltProof<F, C, PCS, FS>
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let commitments =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage1_uni_skip_first_round_proof =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage1_sumcheck_proof =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage2_uni_skip_first_round_proof =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage2_sumcheck_proof =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage3_sumcheck_proof =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage4_sumcheck_proof =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage5_sumcheck_proof =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage6_sumcheck_proof =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage7_sumcheck_proof =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        #[cfg(feature = "zk")]
        let blindfold_proof =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let joint_opening_proof =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        // NOTE: opening_hint is intentionally skipped
        let untrusted_advice_commitment =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        #[cfg(not(feature = "zk"))]
        let opening_claims =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let trace_length: u64 =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let ram_K: u64 =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let rw_config =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let one_hot_config =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let dory_layout =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(JoltProof {
            commitments,
            stage1_uni_skip_first_round_proof,
            stage1_sumcheck_proof,
            stage2_uni_skip_first_round_proof,
            stage2_sumcheck_proof,
            stage3_sumcheck_proof,
            stage4_sumcheck_proof,
            stage5_sumcheck_proof,
            stage6_sumcheck_proof,
            stage7_sumcheck_proof,
            #[cfg(feature = "zk")]
            blindfold_proof,
            joint_opening_proof,
            // NOTE: opening_hint is set to None on deserialization
            opening_hint: None,
            untrusted_advice_commitment,
            #[cfg(not(feature = "zk"))]
            opening_claims,
            trace_length: trace_length as usize,
            ram_K: ram_K as usize,
            rw_config,
            one_hot_config,
            dory_layout,
        })
    }
}

#[cfg(not(feature = "zk"))]
pub struct Claims<F: JoltField>(pub Openings<F>);

#[cfg(not(feature = "zk"))]
impl<F: JoltField> CanonicalSerialize for Claims<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.len().serialize_with_mode(&mut writer, compress)?;
        for (key, (_opening_point, claim)) in self.0.iter() {
            key.serialize_with_mode(&mut writer, compress)?;
            claim.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let mut size = self.0.len().serialized_size(compress);
        for (key, (_opening_point, claim)) in self.0.iter() {
            size += key.serialized_size(compress);
            size += claim.serialized_size(compress);
        }
        size
    }
}

#[cfg(not(feature = "zk"))]
impl<F: JoltField> Valid for Claims<F> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

#[cfg(not(feature = "zk"))]
impl<F: JoltField> CanonicalDeserialize for Claims<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let size = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let mut claims = BTreeMap::new();
        for _ in 0..size {
            let key = OpeningId::deserialize_with_mode(&mut reader, compress, validate)?;
            let claim = F::deserialize_with_mode(&mut reader, compress, validate)?;
            claims.insert(key, (OpeningPoint::default(), claim));
        }
        Ok(Claims(claims))
    }
}

impl CanonicalSerialize for DoryLayout {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        u8::from(*self).serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        u8::from(*self).serialized_size(compress)
    }
}

impl Valid for DoryLayout {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for DoryLayout {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let value = u8::deserialize_with_mode(reader, compress, validate)?;
        if value > 1 {
            return Err(SerializationError::InvalidData);
        }
        Ok(DoryLayout::from(value))
    }
}

// Compact encoding for OpeningId:
// Each variant uses a fused byte = BASE + sumcheck_id (1 byte total for advice, 2 bytes for committed/virtual)
// - [0, NUM_SUMCHECKS) = UntrustedAdvice(sumcheck_id)
// - [NUM_SUMCHECKS, 2*NUM_SUMCHECKS) = TrustedAdvice(sumcheck_id)
// - [2*NUM_SUMCHECKS, 3*NUM_SUMCHECKS) + poly_index = Committed(poly, sumcheck_id)
// - [3*NUM_SUMCHECKS, 4*NUM_SUMCHECKS) + poly_index = Virtual(poly, sumcheck_id)
const OPENING_ID_UNTRUSTED_ADVICE_BASE: u8 = 0;
const OPENING_ID_TRUSTED_ADVICE_BASE: u8 =
    OPENING_ID_UNTRUSTED_ADVICE_BASE + SumcheckId::COUNT as u8;
const OPENING_ID_COMMITTED_BASE: u8 = OPENING_ID_TRUSTED_ADVICE_BASE + SumcheckId::COUNT as u8;
const OPENING_ID_VIRTUAL_BASE: u8 = OPENING_ID_COMMITTED_BASE + SumcheckId::COUNT as u8;

impl CanonicalSerialize for OpeningId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            OpeningId::UntrustedAdvice(sumcheck_id) => {
                let fused = OPENING_ID_UNTRUSTED_ADVICE_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::TrustedAdvice(sumcheck_id) => {
                let fused = OPENING_ID_TRUSTED_ADVICE_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::Polynomial(PolynomialId::Committed(committed_polynomial), sumcheck_id) => {
                let fused = OPENING_ID_COMMITTED_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)?;
                committed_polynomial.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::Polynomial(PolynomialId::Virtual(virtual_polynomial), sumcheck_id) => {
                let fused = OPENING_ID_VIRTUAL_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)?;
                virtual_polynomial.serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            OpeningId::UntrustedAdvice(_) | OpeningId::TrustedAdvice(_) => 1,
            OpeningId::Polynomial(PolynomialId::Committed(committed_polynomial), _) => {
                1 + committed_polynomial.serialized_size(compress)
            }
            OpeningId::Polynomial(PolynomialId::Virtual(virtual_polynomial), _) => {
                1 + virtual_polynomial.serialized_size(compress)
            }
        }
    }
}

impl Valid for OpeningId {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for OpeningId {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let fused = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match fused {
            _ if fused < OPENING_ID_TRUSTED_ADVICE_BASE => {
                let sumcheck_id = fused - OPENING_ID_UNTRUSTED_ADVICE_BASE;
                Ok(OpeningId::UntrustedAdvice(
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ if fused < OPENING_ID_COMMITTED_BASE => {
                let sumcheck_id = fused - OPENING_ID_TRUSTED_ADVICE_BASE;
                Ok(OpeningId::TrustedAdvice(
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ if fused < OPENING_ID_VIRTUAL_BASE => {
                let sumcheck_id = fused - OPENING_ID_COMMITTED_BASE;
                let polynomial =
                    CommittedPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::committed(
                    polynomial,
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ => {
                let sumcheck_id = fused - OPENING_ID_VIRTUAL_BASE;
                let polynomial =
                    VirtualPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::virt(
                    polynomial,
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
        }
    }
}

impl CanonicalSerialize for CommittedPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::RdInc => 0u8.serialize_with_mode(writer, compress),
            Self::RamInc => 1u8.serialize_with_mode(writer, compress),
            Self::InstructionRa(i) => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::BytecodeRa(i) => {
                3u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::RamRa(i) => {
                4u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::TrustedAdvice => 5u8.serialize_with_mode(writer, compress),
            Self::UntrustedAdvice => 6u8.serialize_with_mode(writer, compress),
        }
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        match self {
            Self::RdInc | Self::RamInc | Self::TrustedAdvice | Self::UntrustedAdvice => 1,
            Self::InstructionRa(_) | Self::BytecodeRa(_) | Self::RamRa(_) => 2,
        }
    }
}

impl Valid for CommittedPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for CommittedPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(
            match u8::deserialize_with_mode(&mut reader, compress, validate)? {
                0 => Self::RdInc,
                1 => Self::RamInc,
                2 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::InstructionRa(i as usize)
                }
                3 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::BytecodeRa(i as usize)
                }
                4 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::RamRa(i as usize)
                }
                5 => Self::TrustedAdvice,
                6 => Self::UntrustedAdvice,
                _ => return Err(SerializationError::InvalidData),
            },
        )
    }
}

impl CanonicalSerialize for VirtualPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::PC => 0u8.serialize_with_mode(&mut writer, compress),
            Self::UnexpandedPC => 1u8.serialize_with_mode(&mut writer, compress),
            Self::NextPC => 2u8.serialize_with_mode(&mut writer, compress),
            Self::NextUnexpandedPC => 3u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsNoop => 4u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsVirtual => 5u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsFirstInSequence => 6u8.serialize_with_mode(&mut writer, compress),
            Self::LeftLookupOperand => 7u8.serialize_with_mode(&mut writer, compress),
            Self::RightLookupOperand => 8u8.serialize_with_mode(&mut writer, compress),
            Self::LeftInstructionInput => 9u8.serialize_with_mode(&mut writer, compress),
            Self::RightInstructionInput => 10u8.serialize_with_mode(&mut writer, compress),
            Self::Product => 11u8.serialize_with_mode(&mut writer, compress),
            Self::ShouldJump => 12u8.serialize_with_mode(&mut writer, compress),
            Self::ShouldBranch => 13u8.serialize_with_mode(&mut writer, compress),
            Self::Rd => 14u8.serialize_with_mode(&mut writer, compress),
            Self::Imm => 15u8.serialize_with_mode(&mut writer, compress),
            Self::Rs1Value => 16u8.serialize_with_mode(&mut writer, compress),
            Self::Rs2Value => 17u8.serialize_with_mode(&mut writer, compress),
            Self::RdWriteValue => 18u8.serialize_with_mode(&mut writer, compress),
            Self::Rs1Ra => 19u8.serialize_with_mode(&mut writer, compress),
            Self::Rs2Ra => 20u8.serialize_with_mode(&mut writer, compress),
            Self::RdWa => 21u8.serialize_with_mode(&mut writer, compress),
            Self::LookupOutput => 22u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRaf => 23u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRafFlag => 24u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRa(i) => {
                25u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::RegistersVal => 26u8.serialize_with_mode(&mut writer, compress),
            Self::RamAddress => 27u8.serialize_with_mode(&mut writer, compress),
            Self::RamRa => 28u8.serialize_with_mode(&mut writer, compress),
            Self::RamReadValue => 29u8.serialize_with_mode(&mut writer, compress),
            Self::RamWriteValue => 30u8.serialize_with_mode(&mut writer, compress),
            Self::RamVal => 31u8.serialize_with_mode(&mut writer, compress),
            Self::RamValInit => 32u8.serialize_with_mode(&mut writer, compress),
            Self::RamValFinal => 33u8.serialize_with_mode(&mut writer, compress),
            Self::RamHammingWeight => 34u8.serialize_with_mode(&mut writer, compress),
            Self::UnivariateSkip => 35u8.serialize_with_mode(&mut writer, compress),
            Self::OpFlags(flags) => {
                36u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flags as usize).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::InstructionFlags(flags) => {
                37u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flags as usize).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::LookupTableFlag(flag) => {
                38u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flag).unwrap()).serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        match self {
            Self::PC
            | Self::UnexpandedPC
            | Self::NextPC
            | Self::NextUnexpandedPC
            | Self::NextIsNoop
            | Self::NextIsVirtual
            | Self::NextIsFirstInSequence
            | Self::LeftLookupOperand
            | Self::RightLookupOperand
            | Self::LeftInstructionInput
            | Self::RightInstructionInput
            | Self::Product
            | Self::ShouldJump
            | Self::ShouldBranch
            | Self::Rd
            | Self::Imm
            | Self::Rs1Value
            | Self::Rs2Value
            | Self::RdWriteValue
            | Self::Rs1Ra
            | Self::Rs2Ra
            | Self::RdWa
            | Self::LookupOutput
            | Self::InstructionRaf
            | Self::InstructionRafFlag
            | Self::RegistersVal
            | Self::RamAddress
            | Self::RamRa
            | Self::RamReadValue
            | Self::RamWriteValue
            | Self::RamVal
            | Self::RamValInit
            | Self::RamValFinal
            | Self::RamHammingWeight
            | Self::UnivariateSkip => 1,
            Self::InstructionRa(_)
            | Self::OpFlags(_)
            | Self::InstructionFlags(_)
            | Self::LookupTableFlag(_) => 2,
        }
    }
}

impl Valid for VirtualPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for VirtualPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(
            match u8::deserialize_with_mode(&mut reader, compress, validate)? {
                0 => Self::PC,
                1 => Self::UnexpandedPC,
                2 => Self::NextPC,
                3 => Self::NextUnexpandedPC,
                4 => Self::NextIsNoop,
                5 => Self::NextIsVirtual,
                6 => Self::NextIsFirstInSequence,
                7 => Self::LeftLookupOperand,
                8 => Self::RightLookupOperand,
                9 => Self::LeftInstructionInput,
                10 => Self::RightInstructionInput,
                11 => Self::Product,
                12 => Self::ShouldJump,
                13 => Self::ShouldBranch,
                14 => Self::Rd,
                15 => Self::Imm,
                16 => Self::Rs1Value,
                17 => Self::Rs2Value,
                18 => Self::RdWriteValue,
                19 => Self::Rs1Ra,
                20 => Self::Rs2Ra,
                21 => Self::RdWa,
                22 => Self::LookupOutput,
                23 => Self::InstructionRaf,
                24 => Self::InstructionRafFlag,
                25 => {
                    let i = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::InstructionRa(i as usize)
                }
                26 => Self::RegistersVal,
                27 => Self::RamAddress,
                28 => Self::RamRa,
                29 => Self::RamReadValue,
                30 => Self::RamWriteValue,
                31 => Self::RamVal,
                32 => Self::RamValInit,
                33 => Self::RamValFinal,
                34 => Self::RamHammingWeight,
                35 => Self::UnivariateSkip,
                36 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    let flags = CircuitFlags::from_repr(discriminant)
                        .ok_or(SerializationError::InvalidData)?;
                    Self::OpFlags(flags)
                }
                37 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    let flags = InstructionFlags::from_repr(discriminant)
                        .ok_or(SerializationError::InvalidData)?;
                    Self::InstructionFlags(flags)
                }
                38 => {
                    let flag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::LookupTableFlag(flag as usize)
                }
                _ => return Err(SerializationError::InvalidData),
            },
        )
    }
}

pub fn serialize_and_print_size(
    item_name: &str,
    file_name: &str,
    item: &impl CanonicalSerialize,
) -> Result<(), SerializationError> {
    use std::fs::File;
    let mut file = File::create(file_name)?;
    item.serialize_compressed(&mut file)?;
    let file_size_bytes = file.metadata()?.len();
    let file_size_kb = file_size_bytes as f64 / 1024.0;
    tracing::info!("{item_name} Written to {file_name}");
    tracing::info!("{item_name} size: {file_size_kb:.1} kB");
    Ok(())
}

// ============================================================================
// Batch Proof Aggregation Types
// ============================================================================

/// Data extracted from a JoltProof for batch aggregation.
/// Contains only the data needed for aggregating the Dory opening proof.
///
/// NOTE: True batch aggregation requires prover-side changes to compute
/// a combined RLC polynomial from all transactions together. This structure
/// is for the simpler case of aggregating commitments post-hoc.
pub struct BatchProofData<F, C, PCS, FS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    /// Combined commitment from all polynomial commitments
    pub combined_commitment: PCS::Commitment,
    /// The joint opening proof for this transaction
    pub joint_opening_proof: PCS::Proof,
    /// The claim at the opening point
    pub joint_claim: F,
    /// Hints needed for batch hint combination
    pub opening_hint: Option<PCS::OpeningProofHint>,
    /// Transaction index for gamma power computation
    pub tx_index: usize,
    _phantom: std::marker::PhantomData<(C, FS)>,
}

impl<F, C, PCS, FS> Clone for BatchProofData<F, C, PCS, FS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    fn clone(&self) -> Self {
        Self {
            combined_commitment: self.combined_commitment.clone(),
            joint_opening_proof: self.joint_opening_proof.clone(),
            joint_claim: self.joint_claim,
            opening_hint: self.opening_hint.clone(),
            tx_index: self.tx_index,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F, C, PCS, FS> std::fmt::Debug for BatchProofData<F, C, PCS, FS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchProofData")
            .field("tx_index", &self.tx_index)
            .field("joint_claim", &self.joint_claim)
            .finish()
    }
}

impl<F, C, PCS, FS> BatchProofData<F, C, PCS, FS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    /// Create batch proof data from a JoltProof
    #[cfg(not(feature = "zk"))]
    pub fn from_jolt_proof(proof: &JoltProof<F, C, PCS, FS>, tx_index: usize) -> Self {
        // Use the first commitment as combined commitment placeholder
        // Real implementation requires prover-side changes
        let combined_commitment = proof.commitments.first().cloned().unwrap_or_default();

        // joint_claim extraction requires prover-side changes
        // For now, use zero as placeholder
        let joint_claim = F::zero();

        Self {
            combined_commitment,
            joint_opening_proof: proof.joint_opening_proof.clone(),
            joint_claim,
            opening_hint: proof.opening_hint.clone(),
            tx_index,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute the batch gamma power for this transaction
    pub fn gamma_power(&self, gamma: &F) -> F {
        let mut result = F::one();
        for _ in 0..self.tx_index {
            result = result * *gamma;
        }
        result
    }
}

/// Aggregated batch proof for all transactions in a block.
///
/// For true batch aggregation, the prover must:
/// 1. Compute combined polynomial data from all transactions
/// 2. Build combined RLC polynomial
/// 3. Generate single Dory proof for combined polynomial
///
/// See memory/batch_aggregation_plan.md for full implementation plan.
pub struct BatchBlockProof<F, C, PCS, FS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    /// Combined commitment: sum_i(gamma^i * commitment_i)
    pub combined_commitment: PCS::Commitment,
    /// Single aggregated Dory opening proof (requires prover-side aggregation)
    pub aggregated_opening_proof: PCS::Proof,
    /// Combined claim: sum_i(gamma^i * joint_claim_i)
    pub combined_claim: F,
    /// Gamma challenge used for aggregation
    pub batch_gamma: F,
    /// Number of transactions in the batch
    pub num_transactions: usize,
    /// Individual proof data
    pub proof_data: Vec<BatchProofData<F, C, PCS, FS>>,
}

impl<F, C, PCS, FS> Clone for BatchBlockProof<F, C, PCS, FS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    fn clone(&self) -> Self {
        Self {
            combined_commitment: self.combined_commitment.clone(),
            aggregated_opening_proof: self.aggregated_opening_proof.clone(),
            combined_claim: self.combined_claim,
            batch_gamma: self.batch_gamma,
            num_transactions: self.num_transactions,
            proof_data: self.proof_data.clone(),
        }
    }
}

impl<F, C, PCS, FS> std::fmt::Debug for BatchBlockProof<F, C, PCS, FS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchBlockProof")
            .field("num_transactions", &self.num_transactions)
            .field("combined_claim", &self.combined_claim)
            .finish()
    }
}

impl<F, C, PCS, FS> BatchBlockProof<F, C, PCS, FS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    /// Create a batch block proof from individual transaction proofs
    ///
    /// WARNING: This is a placeholder that does NOT produce a valid
    /// aggregated opening proof. True batch aggregation requires prover-side
    /// changes to compute a combined Dory proof using:
    /// 1. PCS::combine_commitments() for commitment aggregation
    /// 2. PCS::combine_hints() for hint aggregation
    /// 3. Recomputing the Dory proof from combined polynomial data
    pub fn aggregate(proofs: Vec<BatchProofData<F, C, PCS, FS>>, batch_gamma: F) -> Self {
        let num_transactions = proofs.len();
        if proofs.is_empty() {
            panic!("No proofs provided for batch aggregation");
        }

        // Placeholder combined commitment - real impl needs PCS::combine_commitments()
        let combined_commitment = proofs.first().map(|p| p.combined_commitment.clone()).unwrap_or_else(|| PCS::Commitment::default());

        // Compute combined claim from all proofs
        let combined_claim = proofs
            .iter()
            .fold(F::zero(), |acc, p| {
                let gamma_power = p.gamma_power(&batch_gamma);
                acc + (gamma_power * p.joint_claim)
            });

        Self {
            combined_commitment,
            aggregated_opening_proof: proofs[0].joint_opening_proof.clone(),
            combined_claim,
            batch_gamma,
            num_transactions,
            proof_data: proofs,
        }
    }
}
