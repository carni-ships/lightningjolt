#!/bin/bash
# run_phase_benchmark.sh
# Reproduces the prover phase breakdown benchmark
#
# Usage:
#   ./run_phase_benchmark.sh <RPC_URL> <CACHE_DIR> <BLOCK_NUMBER> [NUM_TX]
#
# Features enabled:
#   - Dory-Prime (parallel proof generation)
#   - Lattice NTT commitment (~2x speedup)
#   - Metal GPU acceleration (if available)
#
# Example:
#   ./run_phase_benchmark.sh "https://eth.llamarpc.com" /tmp/ethrex-block-cache 24922263 10

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
RPC_URL="${1:-}"
CACHE_DIR="${2:-/tmp/ethrex-block-cache}"
BLOCK_NUM="${3:-24922263}"
NUM_TX="${4:-10}"

# Features: Dory-Prime (default) + Lattice + Metal GPU
FEATURES="--features lattice,metal-pairing"

# Validate inputs
if [ -z "$RPC_URL" ]; then
    echo -e "${RED}Error: RPC URL required${NC}"
    echo "Usage: $0 <RPC_URL> <CACHE_DIR> <BLOCK_NUMBER> [NUM_TX]"
    echo ""
    echo "Features: Dory-Prime + Lattice NTT + Metal GPU"
    echo "Example:"
    echo "  $0 \"https://eth.llamarpc.com\" /tmp/ethrex-block-cache 24922263 10"
    exit 1
fi

# Check if cache exists
if [ ! -d "$CACHE_DIR" ]; then
    echo -e "${YELLOW}Cache directory not found, creating...${NC}"
    mkdir -p "$CACHE_DIR"
fi

# Check if block is cached
if [ -f "$CACHE_DIR/block_$(printf '%010d' $BLOCK_NUM).json" ]; then
    echo -e "${GREEN}Using cached block #$BLOCK_NUM${NC}"
else
    echo -e "${YELLOW}Fetching block #$BLOCK_NUM from RPC${NC}"
fi

echo "============================================================"
echo "Prover Phase Breakdown Benchmark"
echo "============================================================"
echo "Block: #$BLOCK_NUM"
echo "RPC: $RPC_URL"
echo "Cache: $CACHE_DIR"
echo "Transactions to prove: $NUM_TX"
echo "Features: Dory-Prime + Lattice NTT + Metal GPU"
echo "============================================================"
echo ""

# Run the benchmark with features
START_TIME=$(date +%s)

echo "Running ethrex_realtime_prove with Dory-Prime + Lattice + Metal GPU..."
timeout 1800 cargo run -p ethrex-trace --example ethrex_realtime_prove $FEATURES -- \
    "$RPC_URL" "$CACHE_DIR" "$BLOCK_NUM" 2>&1 | tee /tmp/prover_benchmark_$(date +%Y%m%d_%H%M%S).log

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "Benchmark completed in ${ELAPSED}s"
echo "============================================================"
echo ""
echo "To view detailed phase breakdown, check:"
echo "  cat /tmp/prover_benchmark_*.log | grep PROFILE"
