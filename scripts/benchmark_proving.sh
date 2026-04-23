#!/bin/bash
# Benchmark script for ethrex_block_prove
# Usage: ./scripts/benchmark_proving.sh [block_number] [rpc_url]
#
# Required features:
#   - lattice: Enable NTT-based commitment (~2x speedup)
#   - metal-pairing: Enable GPU-accelerated elliptic curve pairings
#   - parallel: Enable rayon parallel processing

set -e

BLOCK="${1:-21000000}"
RPC="${2:-https://ethereum.publicnode.com}"

export DYLD_LIBRARY_PATH="/Users/carnation/Documents/Claude/lightningjolt/target/release/deps/icicle/lib:$DYLD_LIBRARY_PATH"

echo "=============================================="
echo "Ethereum Block Proving Benchmark"
echo "Block: $BLOCK"
echo "Features: lattice + metal-pairing + parallel"
echo "=============================================="
echo ""

# Build with proper features for GPU acceleration
cargo build -p ethrex-trace --release \
  --features "lattice,metal-pairing" \
  --example ethrex_block_prove

# Run benchmark
./target/release/examples/ethrex_block_prove "$BLOCK" "$RPC"