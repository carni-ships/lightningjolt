mod blake2b;
mod keccak;
mod transcript;

pub use blake2b::Blake2bTranscript;
pub use keccak::KeccakTranscript;
pub use transcript::Transcript;

/// Trait for transcripts that expose their internal state for export/testing.
pub trait HasState {
    fn state_bytes(&self) -> &[u8; 32];
    fn n_rounds_value(&self) -> u32;
}

impl HasState for KeccakTranscript {
    fn state_bytes(&self) -> &[u8; 32] {
        &self.state
    }

    fn n_rounds_value(&self) -> u32 {
        self.n_rounds
    }
}

impl HasState for Blake2bTranscript {
    fn state_bytes(&self) -> &[u8; 32] {
        &self.state
    }

    fn n_rounds_value(&self) -> u32 {
        self.n_rounds
    }
}
