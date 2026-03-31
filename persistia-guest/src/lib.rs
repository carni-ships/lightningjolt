#![cfg_attr(feature = "guest", no_std)]

//! Persistia State Transition Proof — Jolt Guest Program
//!
//! Proves the same core properties as the SP1 guest:
//! 1. Mutations produce the correct SHA-256 Merkle root
//! 2. Quorum threshold is met (signature count >= 2f+1)
//!
//! Differences from SP1:
//! - Ed25519 verification not done in-circuit (no precompile). Instead,
//!   signature validity is passed as a count via advice and the verifier
//!   checks the commitment externally. This isolates the Merkle proving
//!   benchmark from Ed25519 overhead.
//! - No recursive/IVC composition (Jolt doesn't support it yet)
//! - Uses jolt-inlines-sha2 for in-circuit SHA-256

extern crate alloc;

/// Maximum mutations per block (must match state-proofs.ts)
const MAX_MUTATIONS: usize = 64;
/// Maximum leaf data bytes per mutation (key + value)
const _MAX_LEAF_BYTES: usize = 128;

const HEX: [u8; 16] = *b"0123456789abcdef";

/// Compute SHA-256 leaf hash: SHA256("leaf:{key_hex}:{value_hex}")
/// Matches computeLeafHash() in state-proofs.ts
fn sha256_leaf(key: &[u8], key_len: usize, value: &[u8], value_len: usize) -> [u8; 32] {
    // Build the preimage: "leaf:" + hex(key) + ":" + hex(value)
    // Max size: 5 + 128*2 + 1 + 128*2 = 518 bytes
    let mut buf = [0u8; 600];
    let mut pos = 0;

    // "leaf:"
    buf[pos..pos + 5].copy_from_slice(b"leaf:");
    pos += 5;

    // hex(key)
    for i in 0..key_len {
        buf[pos] = HEX[(key[i] >> 4) as usize];
        buf[pos + 1] = HEX[(key[i] & 0xf) as usize];
        pos += 2;
    }

    // ":"
    buf[pos] = b':';
    pos += 1;

    // hex(value)
    for i in 0..value_len {
        buf[pos] = HEX[(value[i] >> 4) as usize];
        buf[pos + 1] = HEX[(value[i] & 0xf) as usize];
        pos += 2;
    }

    jolt_inlines_sha2::Sha256::digest(&buf[..pos])
}

/// Compute SHA-256 node hash: SHA256(hex(left) + hex(right))
fn sha256_node(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 128]; // 32*2 + 32*2 = 128 hex chars
    let mut pos = 0;
    for &b in left.iter() {
        buf[pos] = HEX[(b >> 4) as usize];
        buf[pos + 1] = HEX[(b & 0xf) as usize];
        pos += 2;
    }
    for &b in right.iter() {
        buf[pos] = HEX[(b >> 4) as usize];
        buf[pos + 1] = HEX[(b & 0xf) as usize];
        pos += 2;
    }
    jolt_inlines_sha2::Sha256::digest(&buf)
}

/// Prove a Persistia state transition.
///
/// Inputs:
/// - `mutation_count`: number of active mutations (0..MAX_MUTATIONS)
/// - `keys_flat`: flattened mutation keys, MAX_MUTATIONS * MAX_LEAF_BYTES
/// - `key_lens`: length of each key
/// - `values_flat`: flattened mutation values
/// - `value_lens`: length of each value
/// - `expected_root`: the claimed new state root
/// - `valid_sigs`: count of valid Ed25519 signatures (verified externally)
/// - `active_nodes`: total active validator count
/// - `block_number`: the block/round number
///
/// Output: the verified state root (32 bytes) — proves computation is correct.
#[jolt::provable(heap_size = 131072, stack_size = 65536, max_trace_length = 8388608)]
pub fn prove_block(
    mutation_count: u32,
    keys_flat: &[u8],
    key_lens: &[u8],
    values_flat: &[u8],
    value_lens: &[u8],
    expected_root: [u8; 32],
    valid_sigs: u32,
    active_nodes: u32,
    _block_number: u64,
) -> [u8; 32] {
    let n = mutation_count as usize;

    // 1. Verify quorum: 2f+1 where f = (n-1)/3
    let f = if active_nodes > 0 {
        (active_nodes - 1) / 3
    } else {
        0
    };
    let quorum = 2 * f + 1;
    assert!(valid_sigs >= quorum, "Quorum not met");

    // 2. Compute leaf hashes from mutation data
    let mut leaves = [[0u8; 32]; MAX_MUTATIONS];
    let mut leaf_count = 0usize;
    let mut key_offset = 0usize;
    let mut val_offset = 0usize;

    for i in 0..n {
        let kl = key_lens[i] as usize;
        let vl = value_lens[i] as usize;

        if vl > 0 {
            // Non-delete mutation — compute leaf hash
            leaves[leaf_count] = sha256_leaf(
                &keys_flat[key_offset..key_offset + kl],
                kl,
                &values_flat[val_offset..val_offset + vl],
                vl,
            );
            leaf_count += 1;
        }

        key_offset += kl;
        val_offset += vl;
    }

    // 3. Compute Merkle root bottom-up
    if leaf_count == 0 {
        assert!(
            expected_root == [0u8; 32],
            "Empty mutations should have zero root"
        );
        return expected_root;
    }

    // Pad to power of 2
    let mut size = 1;
    while size < leaf_count {
        size *= 2;
    }
    // leaves already zero-padded beyond leaf_count

    // Bottom-up Merkle computation
    let mut layer = [[0u8; 32]; MAX_MUTATIONS];
    for i in 0..size {
        layer[i] = leaves[i];
    }
    let mut width = size;
    while width > 1 {
        let half = width / 2;
        for i in 0..half {
            layer[i] = sha256_node(&layer[2 * i], &layer[2 * i + 1]);
        }
        width = half;
    }

    let computed_root = layer[0];

    // 4. Assert Merkle root matches expected
    assert!(computed_root == expected_root, "Merkle root mismatch");

    computed_root
}
