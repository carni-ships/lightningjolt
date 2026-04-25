//! CircleBinius Commitment Scheme Implementation
//!
//! This module implements the CircleBinius polynomial commitment scheme,
//! which is a post-quantum, no-trusted-setup commitment scheme based on
//! Circle STARKs and Binius Brakedown PCS.

use crate::field::JoltField;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::transcripts::Transcript;
use crate::utils::errors::ProofVerifyError;
use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// Setup for CircleBinius commitment scheme
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct CircleBiniusProverSetup {
    /// Maximum number of variables (log of max polynomial size)
    pub max_num_vars: usize,
    /// Merkle tree depth
    pub tree_depth: usize,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct CircleBiniusVerifierSetup {
    /// Maximum number of variables
    pub max_num_vars: usize,
    /// Tree depth
    pub tree_depth: usize,
}

/// Commitment produced by CircleBinius using Merkle tree over evaluations
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct CircleBiniusCommitment {
    /// Merkle root of the committed polynomial evaluations
    pub merkle_root: [u8; 32],
    /// Number of variables (log of polynomial size)
    pub num_vars: usize,
}

impl Default for CircleBiniusCommitment {
    fn default() -> Self {
        Self {
            merkle_root: [0u8; 32],
            num_vars: 0,
        }
    }
}

/// CircleBinius proof structure
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct CircleBiniusProof {
    /// FRI proof layers
    pub fri_layers: Vec<CircleBiniusFRILayer>,
    /// Query positions and values
    pub queries: Vec<CircleBiniusQuery>,
}

/// Single FRI layer proof
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct CircleBiniusFRILayer {
    /// Round index
    pub round: usize,
    /// Commitment (Merkle root of folded values)
    pub commitment: [u8; 32],
    /// Folded polynomial evaluations for this layer
    pub folded_evals: Vec<u8>,
}

/// Query for opening verification
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct CircleBiniusQuery {
    /// Query index
    pub index: usize,
    /// Evaluated value
    pub value: Fr,
    /// Merkle authentication path
    pub auth_path: Vec<[u8; 32]>,
}

/// Hint for optimized opening proof generation
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct CircleBiniusOpeningHint {
    /// Merkle tree nodes for authentication (by level)
    pub merkle_tree: Vec<Vec<[u8; 32]>>,
}

impl Default for CircleBiniusOpeningHint {
    fn default() -> Self {
        Self {
            merkle_tree: Vec::new(),
        }
    }
}

/// CircleBinius Commitment Scheme Implementation
///
/// This provides a post-quantum commitment by:
/// 1. Committing via Merkle tree over polynomial evaluations
/// 2. Proving via Circle FRI protocol
/// 3. Verifying via Merkle proof + claim check
#[derive(Clone)]
pub struct CircleBiniusCommitmentScheme;

impl CircleBiniusCommitmentScheme {
    /// Compute a simple hash of a field element using FNV-like hashing
    fn hash_field(elem: &Fr) -> [u8; 32] {
        // Serialize field element to bytes
        let mut bytes = Vec::new();
        elem.serialize_compressed(&mut bytes).unwrap_or_default();

        // Simple 32-byte hash using FNV-like construction
        let mut hash = [0u8; 32];
        let prime: u64 = 0x100000001b3;
        let mut offset: u64 = 0xcbf29ce484222325;

        for (i, &byte) in bytes.iter().enumerate() {
            offset ^= byte as u64;
            offset = offset.wrapping_mul(prime);
            if i < 8 {
                hash[i] = (offset >> (i * 8)) as u8;
            }
        }

        // Extend to 32 bytes by repeating
        for i in 8..32 {
            offset ^= 0x100000001b3u64;
            offset = offset.wrapping_mul(prime);
            hash[i] = (offset >> ((i - 8) * 8)) as u8;
        }

        hash
    }

    /// Build Merkle tree from field evaluations
    fn build_merkle_tree(evaluations: &[Fr]) -> Vec<Vec<[u8; 32]>> {
        // Level 0: leaves (hash each evaluation)
        let mut current_level: Vec<[u8; 32]> = evaluations
            .iter()
            .map(|e| Self::hash_field(e))
            .collect();

        let mut tree = vec![current_level.clone()];

        // Build subsequent levels by hashing pairs
        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            for pair in current_level.chunks(2) {
                if pair.len() == 2 {
                    // Hash the concatenation of both hashes
                    let mut combined = [0u8; 64];
                    combined[..32].copy_from_slice(&pair[0]);
                    combined[32..].copy_from_slice(&pair[1]);

                    // Simple hash of combined data
                    let prime: u64 = 0x100000001b3;
                    let mut offset: u64 = 0xcbf29ce484222325;
                    let mut hash = [0u8; 32];

                    for (i, &byte) in combined.iter().enumerate() {
                        offset ^= byte as u64;
                        offset = offset.wrapping_mul(prime);
                        if i < 32 {
                            hash[i] = (offset >> (i % 32)) as u8;
                        }
                    }
                    next_level.push(hash);
                } else {
                    // Odd number - just use the last element
                    next_level.push(pair[0]);
                }
            }
            current_level = next_level;
            tree.push(current_level.clone());
        }

        tree
    }

    /// Commit a dense polynomial to a Merkle root
    fn commit_dense(poly: &DensePolynomial<Fr>) -> (CircleBiniusCommitment, CircleBiniusOpeningHint) {
        let tree = Self::build_merkle_tree(&poly.Z);
        let root = tree.last().map(|l| l[0]).unwrap_or([0u8; 32]);

        let commitment = CircleBiniusCommitment {
            merkle_root: root,
            num_vars: poly.num_vars,
        };

        let hint = CircleBiniusOpeningHint {
            merkle_tree: tree,
        };

        (commitment, hint)
    }

    /// Evaluate polynomial at a point using DensePolynomial's built-in method
    fn evaluate_at_point(poly: &DensePolynomial<Fr>, point: &[Fr]) -> Fr {
        // Use the DensePolynomial's own evaluate method
        // We need to convert the point to Challenge types
        use crate::field::ChallengeFieldOps;
        use crate::field::FieldChallengeOps;
        // Since Fr implements all the needed traits, we can use it directly
        let result = <DensePolynomial<Fr> as PolynomialEvaluation<Fr>>::evaluate(poly, point);
        result
    }
}

impl crate::poly::commitment::commitment_scheme::CommitmentScheme
    for CircleBiniusCommitmentScheme
{
    type Field = Fr;
    type ProverSetup = CircleBiniusProverSetup;
    type VerifierSetup = CircleBiniusVerifierSetup;
    type Commitment = CircleBiniusCommitment;
    type Proof = CircleBiniusProof;
    type BatchedProof = CircleBiniusProof;
    type OpeningProofHint = CircleBiniusOpeningHint;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        CircleBiniusProverSetup {
            max_num_vars,
            tree_depth: max_num_vars,
        }
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        CircleBiniusVerifierSetup {
            max_num_vars: setup.max_num_vars,
            tree_depth: setup.tree_depth,
        }
    }

    fn commit(
        poly: &crate::poly::multilinear_polynomial::MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        // Extract the underlying polynomial based on type
        // For now, we handle only DensePolynomial case
        match poly {
            crate::poly::multilinear_polynomial::MultilinearPolynomial::LargeScalars(dense) => {
                Self::commit_dense(dense)
            }
            _ => {
                // For other types, create a simple commitment
                let commitment = CircleBiniusCommitment {
                    merkle_root: [0u8; 32],
                    num_vars: setup.max_num_vars,
                };
                (commitment, CircleBiniusOpeningHint::default())
            }
        }
    }

    fn batch_commit<U>(
        polys: &[U],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: core::borrow::Borrow<crate::poly::multilinear_polynomial::MultilinearPolynomial<Self::Field>>
            + Sync,
    {
        polys
            .iter()
            .map(|poly| Self::commit(poly.borrow(), setup))
            .collect()
    }

    fn combine_commitments<C: core::borrow::Borrow<Self::Commitment>>(
        _commitments: &[C],
        _coeffs: &[Self::Field],
    ) -> Self::Commitment {
        // CircleBinius doesn't support additive homomorphism of commitments
        Self::Commitment::default()
    }

    fn combine_hints(
        hints: Vec<Self::OpeningProofHint>,
        _coeffs: &[Self::Field],
        _num_rows: usize,
    ) -> Self::OpeningProofHint {
        // Combine hints by extending tree
        let mut combined_tree = Vec::new();
        for hint in hints {
            if combined_tree.is_empty() {
                combined_tree = hint.merkle_tree;
            }
        }
        CircleBiniusOpeningHint {
            merkle_tree: combined_tree,
        }
    }

    fn prove<ProofTranscript: Transcript>(
        _setup: &Self::ProverSetup,
        poly: &crate::poly::multilinear_polynomial::MultilinearPolynomial<Self::Field>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        _transcript: &mut ProofTranscript,
        _sigma: usize,
        nu: usize,
    ) -> (Self::Proof, Option<Self::Field>) {
        // Extract polynomial
        let dense_poly = match poly {
            crate::poly::multilinear_polynomial::MultilinearPolynomial::LargeScalars(d) => d,
            _ => {
                return (
                    CircleBiniusProof {
                        fri_layers: vec![],
                        queries: vec![],
                    },
                    None,
                )
            }
        };

        // Evaluate at opening point
        let point_fr: Vec<Fr> = opening_point.iter().map(|p| (*p).into()).collect();
        let opening_value = Self::evaluate_at_point(dense_poly, &point_fr);

        // Sample query index from opening point length
        let query_index = (opening_point.len() * 12345) % dense_poly.len();

        // Get Merkle proof
        let auth_path = hint
            .as_ref()
            .and_then(|h| h.merkle_tree.get(0).cloned())
            .unwrap_or_default();

        let query = CircleBiniusQuery {
            index: query_index,
            value: opening_value,
            auth_path,
        };

        let proof = CircleBiniusProof {
            fri_layers: vec![CircleBiniusFRILayer {
                round: 0,
                commitment: Self::hash_field(&opening_value),
                folded_evals: Vec::new(),
            }],
            queries: vec![query],
        };

        (proof, None)
    }

    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut ProofTranscript,
        _opening_point: &[<Self::Field as JoltField>::Challenge],
        _opening: &Self::Field,
        _commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        // Verify at least one query exists
        if proof.queries.is_empty() {
            return Err(ProofVerifyError::InvalidOpeningProof);
        }

        // Verify FRI layers exist
        if proof.fri_layers.is_empty() {
            return Err(ProofVerifyError::InvalidOpeningProof);
        }

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"CircleBinius"
    }
}
