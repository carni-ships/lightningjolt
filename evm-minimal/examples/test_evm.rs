//! Simple test binary for evm-minimal
//!
//! Run with: cargo run -p evm-minimal --example test_evm

use evm_minimal::{keccak256, H256, Address, DefaultCrypto};

fn main() {
    println!("Testing evm-minimal crypto operations...\n");

    // Test 1: keccak256
    println!("Test 1: keccak256");
    let input = b"hello world";
    let hash = keccak256(input);
    println!("  Input: {:?}", input);
    println!("  Hash:  0x{}", hex::encode(hash.0));
    println!("  ✓ keccak256 computed successfully\n");

    // Test 2: Type conversions
    println!("Test 2: Type conversions");

    // Address from bytes
    let addr_bytes = [0x12u8; 20];
    let addr = Address(addr_bytes);
    println!("  Address: {}", addr);

    // H256 from bytes
    let hash_bytes = [0xabu8; 32];
    let h256 = H256(hash_bytes);
    println!("  H256:    0x{}", hex::encode(h256.0));

    // H256 to/from U256
    let num = h256.to_uint();
    let back = H256::from_uint(&num);
    assert_eq!(h256.0, back.0, "H256 roundtrip failed!");
    println!("  ✓ H256 <-> U256 roundtrip works\n");

    println!("All tests passed!");
}
