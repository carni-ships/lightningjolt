use super::transcript::Transcript;
use crate::field::JoltField;
use sha3::{Digest, Keccak256};

/// Represents the current state of the protocol's Fiat-Shamir transcript.
#[derive(Default, Clone)]
pub struct KeccakTranscript {
    /// Ethereum-compatible 256-bit running state
    pub state: [u8; 32],
    /// We append an ordinal to each invocation of the hash
    pub(crate) n_rounds: u32,
    #[cfg(test)]
    /// A complete history of the transcript's `state`; used for testing.
    state_history: Vec<[u8; 32]>,
    #[cfg(test)]
    /// For a proof to be valid, the verifier's `state_history` should always match
    /// the prover's. In testing, the Jolt verifier may be provided the prover's
    /// `state_history` so that we can detect any deviations and the backtrace can
    /// tell us where it happened.
    expected_state_history: Option<Vec<[u8; 32]>>,
}

impl KeccakTranscript {
    /// Gives the hasher object with the running seed and index added
    /// To load hash you must call finalize, after appending u8 vectors
    fn hasher(&self) -> Keccak256 {
        let mut packed = [0u8; 32];
        packed[28..].copy_from_slice(&self.n_rounds.to_be_bytes());
        Keccak256::new()
            .chain_update(self.state)
            .chain_update(packed)
    }

    // Loads arbitrary byte lengths using ceil(out/32) invocations of 32 byte randoms
    // Discards top bits when the size is less than 32 bytes
    fn challenge_bytes(&mut self, out: &mut [u8]) {
        let mut remaining_len = out.len();
        let mut start = 0;
        while remaining_len > 32 {
            self.challenge_bytes32(&mut out[start..start + 32]);
            start += 32;
            remaining_len -= 32;
        }
        // We load a full 32 byte random region
        let mut full_rand = [0u8; 32];
        self.challenge_bytes32(&mut full_rand);
        // Then only clone the first bits of this random region to perfectly fill out
        out[start..start + remaining_len].clone_from_slice(&full_rand[0..remaining_len]);
    }

    // Loads exactly 32 bytes from the transcript by hashing the seed with the round constant
    fn challenge_bytes32(&mut self, out: &mut [u8]) {
        assert_eq!(32, out.len());
        let rand: [u8; 32] = self.hasher().finalize().into();
        out.clone_from_slice(rand.as_slice());
        self.update_state(rand);
    }

    fn update_state(&mut self, new_state: [u8; 32]) {
        self.state = new_state;
        self.n_rounds += 1;
        #[cfg(test)]
        {
            if let Some(expected_state_history) = &self.expected_state_history {
                assert!(
                    new_state == expected_state_history[self.n_rounds as usize],
                    "Fiat-Shamir transcript mismatch"
                );
            }
            self.state_history.push(new_state);
        }
    }
}

impl Transcript for KeccakTranscript {
    fn new(label: &'static [u8]) -> Self {
        // Hash in the label
        assert!(label.len() < 33);
        let mut padded = [0u8; 32];
        padded[..label.len()].copy_from_slice(label);
        let out = Keccak256::new().chain_update(padded).finalize();

        Self {
            state: out.into(),
            n_rounds: 0,
            #[cfg(test)]
            state_history: vec![out.into()],
            #[cfg(test)]
            expected_state_history: None,
        }
    }

    #[cfg(test)]
    /// Compare this transcript to `other` and panic if/when they deviate.
    /// Typically used to compare the verifier's transcript to the prover's.
    fn compare_to(&mut self, other: Self) {
        self.expected_state_history = Some(other.state_history);
    }

    // === Internal raw methods (EVM-compatible serialization) ===

    fn raw_append_label(&mut self, label: &'static [u8]) {
        // Labels must fit into one EVM word, right-padded with zeros
        // (matches Solidity's bytes32 string casting)
        assert!(label.len() < 33);
        let mut padded = [0u8; 32];
        padded[..label.len()].copy_from_slice(label);
        let hasher = self.hasher().chain_update(padded);
        self.update_state(hasher.finalize().into());
    }

    fn raw_append_bytes(&mut self, bytes: &[u8]) {
        // Add the message and label
        let hasher = self.hasher().chain_update(bytes);
        self.update_state(hasher.finalize().into());
    }

    fn raw_append_u64(&mut self, x: u64) {
        // Allocate into a 32 byte region (left-padded for EVM uint256 compatibility)
        let mut packed = [0u8; 32];
        packed[24..].copy_from_slice(&x.to_be_bytes());
        let hasher = self.hasher().chain_update(packed);
        self.update_state(hasher.finalize().into());
    }

    fn raw_append_scalar<F: JoltField>(&mut self, scalar: &F) {
        let mut buf = vec![];
        scalar.serialize_uncompressed(&mut buf).unwrap();
        // Serialize uncompressed gives the scalar in LE byte order which is not
        // a natural representation in the EVM for scalar math so we reverse
        // to get an EVM compatible version.
        buf.reverse();
        self.raw_append_bytes(&buf);
    }

    // === Challenge generation methods ===

    fn challenge_u128(&mut self) -> u128 {
        let mut buf = [0u8; 16];
        self.challenge_bytes(&mut buf);
        buf.reverse();
        u128::from_be_bytes(buf)
    }

    fn challenge_scalar<F: JoltField>(&mut self) -> F {
        // Under the hood all Fr are 128 bits for performance
        self.challenge_scalar_128_bits()
    }

    fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F {
        let mut buf = [0u8; 16];
        self.challenge_bytes(&mut buf);
        buf.reverse();
        F::from_bytes(&buf)
    }

    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        (0..len)
            .map(|_i| self.challenge_scalar())
            .collect::<Vec<F>>()
    }

    // Compute powers of scalar q : (1, q, q^2, ..., q^(len-1))
    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let q: F = self.challenge_scalar();
        let mut q_powers = vec![F::one(); len];
        for i in 1..len {
            q_powers[i] = q_powers[i - 1] * q;
        }
        q_powers
    }

    // New methods that return F::Challenge
    fn challenge_scalar_optimized<F: JoltField>(&mut self) -> F::Challenge {
        let mut buf = [0u8; 16];
        self.challenge_bytes(&mut buf);
        buf.reverse();
        F::Challenge::from(u128::from_be_bytes(buf))
    }

    fn challenge_vector_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F::Challenge> {
        (0..len)
            .map(|_i| self.challenge_scalar_optimized::<F>())
            .collect::<Vec<F::Challenge>>()
    }

    fn challenge_scalar_powers_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let q: F::Challenge = self.challenge_scalar_optimized::<F>();
        let q_f: F = q.into();
        let mut q_powers = vec![<F as ark_std::One>::one(); len];
        for i in 1..len {
            q_powers[i] = q_f * q_powers[i - 1];
        }
        q_powers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use std::collections::HashSet;

    fn to_hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn test_solidity_transcript_vectors() {
        use ark_serialize::CanonicalSerialize;

        let mut t = KeccakTranscript::new(b"jolt_v1");
        println!("=== SOLIDITY TRANSCRIPT TEST VECTORS ===");
        println!(
            "after new(\"jolt_v1\"): state=0x{} nRounds={}",
            to_hex(&t.state),
            t.n_rounds
        );

        t.append_label(b"stage");
        println!(
            "after append_label(\"stage\"): state=0x{} nRounds={}",
            to_hex(&t.state),
            t.n_rounds
        );

        t.append_u64(b"trace_len", 1024u64);
        println!(
            "after append_u64(\"trace_len\",1024): state=0x{} nRounds={}",
            to_hex(&t.state),
            t.n_rounds
        );

        let scalar = Fr::from(42u64);
        t.append_scalar(b"claim", &scalar);
        println!(
            "after append_scalar(\"claim\",42): state=0x{} nRounds={}",
            to_hex(&t.state),
            t.n_rounds
        );

        let challenge: Fr = t.challenge_scalar();
        let mut buf = vec![];
        challenge.serialize_uncompressed(&mut buf).unwrap();
        buf.reverse();
        println!(
            "challenge_scalar(): value=0x{} state=0x{} nRounds={}",
            to_hex(&buf),
            to_hex(&t.state),
            t.n_rounds
        );

        let scalars_vec: Vec<Fr> = vec![Fr::from(100u64), Fr::from(200u64), Fr::from(300u64)];
        t.append_scalars(b"coeffs", &scalars_vec);
        println!(
            "after append_scalars(\"coeffs\",[100,200,300]): state=0x{} nRounds={}",
            to_hex(&t.state),
            t.n_rounds
        );

        let powers: Vec<Fr> = t.challenge_scalar_powers(3);
        for (i, p) in powers.iter().enumerate() {
            let mut buf = vec![];
            p.serialize_uncompressed(&mut buf).unwrap();
            buf.reverse();
            println!("powers[{i}]=0x{}", to_hex(&buf));
        }
        println!(
            "after challenge_scalar_powers(3): state=0x{} nRounds={}",
            to_hex(&t.state),
            t.n_rounds
        );
        println!("=== END TEST VECTORS ===");
    }

    #[test]
    fn test_sumcheck_verify_vectors() {
        use crate::poly::unipoly::CompressedUniPoly;
        use crate::subprotocols::sumcheck::ClearSumcheckProof;
        use ark_serialize::CanonicalSerialize;

        // Build a 3-round sumcheck proof with degree-2 polynomials.
        // Each CompressedUniPoly stores [c0, c2] (linear term c1 is recovered from hint).
        //
        // Round 0: p0(x) = 5 + 3x + 2x^2 → coeffs_except_linear = [5, 2]
        //   hint = p0(0) + p0(1) = 5 + (5+3+2) = 15
        //
        // Round 1: p1(x) = 7 + 1x + 4x^2 → coeffs_except_linear = [7, 4]
        //   hint = p1(0) + p1(1) = 7 + (7+1+4) = 19
        //   but actually hint comes from eval_from_hint of previous round
        //
        // Round 2: p2(x) = 3 + 6x + 1x^2 → coeffs_except_linear = [3, 1]

        let polys = vec![
            CompressedUniPoly {
                coeffs_except_linear_term: vec![Fr::from(5u64), Fr::from(2u64)],
            },
            CompressedUniPoly {
                coeffs_except_linear_term: vec![Fr::from(7u64), Fr::from(4u64)],
            },
            CompressedUniPoly {
                coeffs_except_linear_term: vec![Fr::from(3u64), Fr::from(1u64)],
            },
        ];

        // Initial claim = p0(0) + p0(1) = 5 + (5+3+2) = 15
        // But we need to compute the actual claim based on what the prover would produce.
        // For a standalone verify test, claim = f(0) + f(1) for the first polynomial.
        // p0(x) = 5 + 3x + 2x^2 → p0(0) + p0(1) = 5 + 10 = 15
        let initial_claim = Fr::from(15u64);

        let proof: ClearSumcheckProof<Fr, KeccakTranscript> = ClearSumcheckProof::new(polys);

        // Run verify with a known transcript
        let mut transcript = KeccakTranscript::new(b"sumcheck_test");

        let result = proof.verify(initial_claim, 3, 2, &mut transcript);
        let (final_claim, challenges) = result.unwrap();

        // Print test vectors
        println!("=== SUMCHECK VERIFY TEST VECTORS ===");
        println!("initial_claim = {}", 15u64);

        // Print initial transcript state
        let mut t2 = KeccakTranscript::new(b"sumcheck_test");
        println!(
            "transcript_init: state=0x{} nRounds={}",
            to_hex(&t2.state),
            t2.n_rounds
        );

        // Replay round-by-round to get intermediate states
        let _hint = initial_claim;
        let coeffs_sets: Vec<Vec<Fr>> = vec![
            vec![Fr::from(5u64), Fr::from(2u64)],
            vec![Fr::from(7u64), Fr::from(4u64)],
            vec![Fr::from(3u64), Fr::from(1u64)],
        ];

        for (round, coeffs) in coeffs_sets.iter().enumerate() {
            // Append compressed poly to transcript
            t2.append_scalars(b"sumcheck_poly", coeffs);
            println!(
                "round {round} after append_scalars: state=0x{} nRounds={}",
                to_hex(&t2.state),
                t2.n_rounds
            );

            let challenge: Fr = t2.challenge_scalar();
            let mut buf = vec![];
            challenge.serialize_uncompressed(&mut buf).unwrap();
            buf.reverse();
            println!("round {round} challenge: 0x{}", to_hex(&buf));
            println!(
                "round {round} after challenge: state=0x{} nRounds={}",
                to_hex(&t2.state),
                t2.n_rounds
            );
        }

        // Print final claim (as Fr)
        let mut buf = vec![];
        final_claim.serialize_uncompressed(&mut buf).unwrap();
        buf.reverse();
        println!("final_claim = 0x{}", to_hex(&buf));

        // Print challenges as field elements (for Solidity verification)
        for (i, c) in challenges.iter().enumerate() {
            let c_f: Fr = (*c).into();
            let mut buf = vec![];
            c_f.serialize_uncompressed(&mut buf).unwrap();
            buf.reverse();
            println!("challenges_fr[{i}] = 0x{}", to_hex(&buf));
        }
        println!(
            "after verify: state=0x{} nRounds={}",
            to_hex(&transcript.state),
            transcript.n_rounds
        );
        // Verify the relationship: challenge_scalar_optimized field value = raw_u128 * 2^(-128) mod R
        // This is because from_bigint_unchecked stores the value directly in Montgomery form.
        let mut t3 = KeccakTranscript::new(b"relate_test");
        let co: <Fr as JoltField>::Challenge = t3.challenge_scalar_optimized::<Fr>();
        let co_fr: Fr = co.into();
        let raw_u128_value = (co.low as u128) | ((co.high as u128) << 64);

        // Compute 2^(-128) mod R via field inverse
        let two_128 = Fr::from(1u64).mul_pow_2(128);
        let inv_two_128 = two_128.inverse().unwrap();
        let reconstructed =
            Fr::from(raw_u128_value as u64) + Fr::from((raw_u128_value >> 64) as u64).mul_pow_2(64);
        let expected = reconstructed * inv_two_128;

        let mut buf = vec![];
        co_fr.serialize_uncompressed(&mut buf).unwrap();
        buf.reverse();
        println!("challenge_scalar_optimized: 0x{}", to_hex(&buf));

        buf.clear();
        expected.serialize_uncompressed(&mut buf).unwrap();
        buf.reverse();
        println!("raw * 2^(-128) mod R:       0x{}", to_hex(&buf));
        assert_eq!(
            co_fr, expected,
            "from_bigint_unchecked relationship mismatch"
        );
        println!("from_bigint_unchecked = raw * 2^(-128) mod R CONFIRMED");

        // Print the constant 2^(-128) mod R for Solidity
        buf.clear();
        inv_two_128.serialize_uncompressed(&mut buf).unwrap();
        buf.reverse();
        println!("INV_TWO_128 = 0x{}", to_hex(&buf));

        // Also check that both challenge functions produce the same transcript state
        let mut t4 = KeccakTranscript::new(b"relate_test");
        let _cs: Fr = t4.challenge_scalar();
        assert_eq!(t3.state, t4.state, "transcript state mismatch");
        println!("Transcript states match between challenge_scalar and challenge_scalar_optimized");
        println!("=== END SUMCHECK VECTORS ===");
    }

    #[test]
    fn test_batched_sumcheck_verify_vectors() {
        use crate::poly::unipoly::CompressedUniPoly;
        use crate::subprotocols::sumcheck::ClearSumcheckProof;
        use ark_serialize::CanonicalSerialize;

        // Simulate a batched sumcheck with 2 instances, both 3 rounds, degree 2.
        // Instance A: claim_a = 15, Instance B: claim_b = 20
        //
        // The batched verify flow:
        //   1. append_scalar("sumcheck_claim", claim_a)
        //   2. append_scalar("sumcheck_claim", claim_b)
        //   3. challenge_vector(2) → [gamma_a, gamma_b] (non-Montgomery)
        //   4. combined_claim = gamma_a * claim_a + gamma_b * claim_b
        //   5. Run single sumcheck verify on combined_claim with 3 rounds

        let claim_a = Fr::from(15u64);
        let claim_b = Fr::from(20u64);

        let mut transcript = KeccakTranscript::new(b"batched_sc_test");

        // Step 1-2: append claims
        transcript.append_scalar(b"sumcheck_claim", &claim_a);
        transcript.append_scalar(b"sumcheck_claim", &claim_b);

        // Step 3: derive batching coefficients (non-Montgomery)
        let gammas: Vec<Fr> = transcript.challenge_vector(2);

        // Step 4: combined claim
        let combined_claim = gammas[0] * claim_a + gammas[1] * claim_b;

        // Build proof with known polynomials (these are the "combined" polynomials)
        let polys = vec![
            CompressedUniPoly {
                coeffs_except_linear_term: vec![Fr::from(8u64), Fr::from(3u64)],
            },
            CompressedUniPoly {
                coeffs_except_linear_term: vec![Fr::from(11u64), Fr::from(5u64)],
            },
            CompressedUniPoly {
                coeffs_except_linear_term: vec![Fr::from(4u64), Fr::from(2u64)],
            },
        ];

        // For the proof to pass verify, round 0's p(0)+p(1) must equal combined_claim.
        // p0(x) = 8 + linear*x + 3*x^2, p0(0)+p0(1) = 8 + (8+linear+3) = 2*8+linear+3
        // So we need 2*8+linear+3 = combined_claim → linear = combined_claim - 19
        // The proof verifier just checks the sumcheck equation, so we build
        // polynomials whose p(0)+p(1) we control.
        //
        // Actually, for testing the batched flow, we don't need the proof to be
        // mathematically valid — we just need to verify that Solidity produces
        // the same transcript state and challenges. The verify function will still
        // run through all rounds computing eval_from_hint.
        //
        // But ClearSumcheckProof::verify doesn't check p(0)+p(1)==claim for
        // intermediate rounds — it just uses the claim as the hint and keeps going.
        // So we can use any polynomials as long as the initial claim is consistent.

        let proof: ClearSumcheckProof<Fr, KeccakTranscript> = ClearSumcheckProof::new(polys);
        let result = proof.verify(combined_claim, 3, 2, &mut transcript);
        let (final_claim, challenges) = result.unwrap();

        // Print test vectors for Solidity
        println!("=== BATCHED SUMCHECK TEST VECTORS ===");
        println!("claim_a = 15");
        println!("claim_b = 20");

        for (i, g) in gammas.iter().enumerate() {
            let mut buf = vec![];
            g.serialize_uncompressed(&mut buf).unwrap();
            buf.reverse();
            println!("gamma[{i}] = 0x{}", to_hex(&buf));
        }

        let mut buf = vec![];
        combined_claim.serialize_uncompressed(&mut buf).unwrap();
        buf.reverse();
        println!("combined_claim = 0x{}", to_hex(&buf));

        for (i, c) in challenges.iter().enumerate() {
            let c_f: Fr = (*c).into();
            let mut buf = vec![];
            c_f.serialize_uncompressed(&mut buf).unwrap();
            buf.reverse();
            println!("challenges[{i}] = 0x{}", to_hex(&buf));
        }

        buf.clear();
        final_claim.serialize_uncompressed(&mut buf).unwrap();
        buf.reverse();
        println!("final_claim = 0x{}", to_hex(&buf));
        println!(
            "transcript state = 0x{} nRounds = {}",
            to_hex(&transcript.state),
            transcript.n_rounds
        );
        println!("=== END BATCHED VECTORS ===");
    }

    #[test]
    fn test_append_serializable_vectors() {
        use ark_serialize::CanonicalSerialize;

        // Test append_serializable with a known BN254 Fr scalar (simpler than GT element)
        // to validate the serialization format matches Solidity's appendLabeledBytes.
        let mut t = KeccakTranscript::new(b"serial_test");

        // append_serializable(b"commitment", Fr(42)):
        //   1. serialize_uncompressed(42) → 32 bytes LE
        //   2. raw_append_label_with_len(b"commitment", 32)
        //   3. reverse bytes → BE
        //   4. raw_append_bytes(reversed)
        let scalar = Fr::from(42u64);
        let mut buf = vec![];
        scalar.serialize_uncompressed(&mut buf).unwrap();
        let data_len = buf.len() as u64;
        buf.reverse(); // LE to BE
        println!("=== APPEND_SERIALIZABLE VECTORS ===");
        println!("data_len = {data_len}");
        println!("serialized_be = 0x{}", to_hex(&buf));

        t.append_serializable(b"commitment", &scalar);
        println!(
            "after append_serializable: state=0x{} nRounds={}",
            to_hex(&t.state),
            t.n_rounds
        );

        // Also test with a pair of scalars to match commitment-like data
        let s1 = Fr::from(100u64);
        let s2 = Fr::from(200u64);
        t.append_serializable(b"commitment", &(s1, s2));

        let mut buf2 = vec![];
        (s1, s2).serialize_uncompressed(&mut buf2).unwrap();
        println!(
            "pair_data_len = {} pair_serialized_be = 0x{}",
            buf2.len(),
            to_hex(&{
                let mut r = buf2.clone();
                r.reverse();
                r
            })
        );
        println!(
            "after pair: state=0x{} nRounds={}",
            to_hex(&t.state),
            t.n_rounds
        );
        println!("=== END APPEND_SERIALIZABLE VECTORS ===");
    }

    #[test]
    fn test_preamble_vectors() {
        // Generate transcript state after a simplified preamble for Solidity verification.
        // Uses the same operations as fiat_shamir_preamble() in zkvm/mod.rs.
        let mut t = KeccakTranscript::new(b"jolt_v1");
        println!("=== PREAMBLE TEST VECTORS ===");
        println!(
            "after new: state=0x{} nRounds={}",
            to_hex(&t.state),
            t.n_rounds
        );

        // Simulate fiat_shamir_preamble with known values
        t.append_u64(b"max_input_size", 4096);
        t.append_u64(b"max_output_size", 4096);
        t.append_u64(b"heap_size", 65536);
        // inputs: [9, 5, 3] serialized via postcard
        // For testing, use simple raw bytes
        t.append_bytes(b"inputs", &[9, 0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 0]);
        t.append_bytes(b"outputs", &[]);
        t.append_u64(b"panic", 0);
        t.append_u64(b"ram_K", 65536);
        t.append_u64(b"trace_length", 1024);
        t.append_u64(b"entry_address", 0x80000000);

        println!(
            "after preamble: state=0x{} nRounds={}",
            to_hex(&t.state),
            t.n_rounds
        );
        println!("=== END PREAMBLE VECTORS ===");
    }

    #[test]
    fn test_eq_polynomial_vectors() {
        use crate::poly::eq_plus_one_poly::EqPlusOnePolynomial;
        use crate::poly::eq_poly::EqPolynomial;
        use ark_serialize::CanonicalSerialize;

        type Challenge = <Fr as JoltField>::Challenge;

        fn fr_hex(f: &Fr) -> String {
            let mut buf = Vec::new();
            f.serialize_uncompressed(&mut buf).unwrap();
            buf.reverse();
            format!(
                "0x{}",
                buf.iter().map(|b| format!("{b:02x}")).collect::<String>()
            )
        }

        fn ch_hex(c: &Challenge) -> String {
            let f: Fr = (*c).into();
            fr_hex(&f)
        }

        println!("=== EQ POLYNOMIAL VECTORS ===");

        // Test eq(x, y) with field element inputs
        let x: Vec<Fr> = vec![Fr::from(3u64), Fr::from(7u64)];
        let y: Vec<Fr> = vec![Fr::from(11u64), Fr::from(13u64)];
        let eq_val: Fr = EqPolynomial::mle(&x, &y);
        println!("eq([3,7], [11,13]) = {}", fr_hex(&eq_val));

        // Test eq with single var
        let x1: Vec<Fr> = vec![Fr::from(100u64)];
        let y1: Vec<Fr> = vec![Fr::from(200u64)];
        let eq_val1: Fr = EqPolynomial::mle(&x1, &y1);
        println!("eq([100], [200]) = {}", fr_hex(&eq_val1));

        // Test eqPlusOne with transcript-derived challenges
        let mut t = KeccakTranscript::new(b"eq_test");
        let c1: Fr = t.challenge_scalar();
        let c2: Fr = t.challenge_scalar();
        let c3: Fr = t.challenge_scalar();
        let c4: Fr = t.challenge_scalar();

        println!("c1 = {}", fr_hex(&c1));
        println!("c2 = {}", fr_hex(&c2));
        println!("c3 = {}", fr_hex(&c3));
        println!("c4 = {}", fr_hex(&c4));

        // eq(c1, c3) and eq(c2, c4) as Fr
        let eq_c1c3: Fr = EqPolynomial::mle(&[c1], &[c3]);
        println!("eq([c1], [c3]) = {}", fr_hex(&eq_c1c3));

        // eqPlusOne using challenge_scalar_optimized values
        let mut t2 = KeccakTranscript::new(b"eqpo_test");
        let ch1: Challenge = t2.challenge_scalar_optimized::<Fr>();
        let ch2: Challenge = t2.challenge_scalar_optimized::<Fr>();
        let ch3: Challenge = t2.challenge_scalar_optimized::<Fr>();
        let ch4: Challenge = t2.challenge_scalar_optimized::<Fr>();

        println!("ch1 = {} (mont)", ch_hex(&ch1));
        println!("ch2 = {} (mont)", ch_hex(&ch2));
        println!("ch3 = {} (mont)", ch_hex(&ch3));
        println!("ch4 = {} (mont)", ch_hex(&ch4));

        let eqpo_val: Fr = EqPlusOnePolynomial::<Fr>::new(vec![ch1, ch2]).evaluate(&[ch3, ch4]);
        println!("eqPlusOne([ch1,ch2], [ch3,ch4]) = {}", fr_hex(&eqpo_val));

        println!("=== END EQ POLYNOMIAL VECTORS ===");
    }

    #[test]
    fn test_lt_polynomial_vectors() {
        use ark_ff::{One, Zero};
        use ark_serialize::CanonicalSerialize;
        use std::iter::zip;

        type Challenge = <Fr as JoltField>::Challenge;

        fn fr_hex(f: &Fr) -> String {
            let mut buf = Vec::new();
            f.serialize_uncompressed(&mut buf).unwrap();
            buf.reverse();
            format!(
                "0x{}",
                buf.iter().map(|b| format!("{b:02x}")).collect::<String>()
            )
        }

        // Direct LT evaluation: LT(x, y) = Σ_i (1-x_i)*y_i * eq(x[i+1:], y[i+1:])
        fn lt_eval(x: &[Fr], y: &[Fr]) -> Fr {
            let mut result = Fr::zero();
            let mut eq_term = Fr::one();
            for (xi, yi) in zip(x, y) {
                result += (Fr::one() - xi) * yi * eq_term;
                eq_term *= Fr::one() - xi - yi + *xi * *yi + *xi * *yi;
            }
            result
        }

        println!("=== LT POLYNOMIAL VECTORS ===");

        // Generate challenge values from transcript
        let mut t = KeccakTranscript::new(b"lt_test");
        let ch1: Challenge = t.challenge_scalar_optimized::<Fr>();
        let ch2: Challenge = t.challenge_scalar_optimized::<Fr>();
        let ch3: Challenge = t.challenge_scalar_optimized::<Fr>();
        let ch4: Challenge = t.challenge_scalar_optimized::<Fr>();

        let x: Vec<Fr> = vec![ch1.into(), ch2.into()];
        let y: Vec<Fr> = vec![ch3.into(), ch4.into()];

        println!("x[0] = {}", fr_hex(&x[0]));
        println!("x[1] = {}", fr_hex(&x[1]));
        println!("y[0] = {}", fr_hex(&y[0]));
        println!("y[1] = {}", fr_hex(&y[1]));

        let lt_val = lt_eval(&x, &y);
        println!("LT([x0,x1], [y0,y1]) = {}", fr_hex(&lt_val));

        // Also test with Mont challenges directly
        let ch5: Challenge = t.challenge_scalar_optimized::<Fr>();
        let ch6: Challenge = t.challenge_scalar_optimized::<Fr>();
        let x2: Vec<Fr> = vec![ch5.into()];
        let y2: Vec<Fr> = vec![ch6.into()];
        println!("x_single = {}", fr_hex(&x2[0]));
        println!("y_single = {}", fr_hex(&y2[0]));
        let lt_single = lt_eval(&x2, &y2);
        println!("LT([x_single], [y_single]) = {}", fr_hex(&lt_single));

        println!("=== END LT POLYNOMIAL VECTORS ===");
    }

    #[test]
    fn test_challenge_scalar_128_bits() {
        let mut transcript = KeccakTranscript::new(b"test_128_bit_scalar");
        let mut scalars = HashSet::new();

        for i in 0..10000 {
            let scalar: Fr = transcript.challenge_scalar_128_bits();

            let num_bits = scalar.num_bits();
            assert!(
                num_bits <= 128,
                "Scalar at iteration {i} has {num_bits} bits, expected <= 128",
            );

            assert!(
                scalars.insert(scalar),
                "Duplicate scalar found at iteration {i}",
            );
        }
    }
}
