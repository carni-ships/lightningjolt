#!/bin/bash
# Profile-Guided Optimization build for Jolt prover
# Usage: ./scripts/pgo-build.sh [benchmark_name] [--icicle]
# Default benchmark: sha3
set -euo pipefail

BENCH="${1:-sha3}"
FEATURES="host"
ICICLE_ENV=""

# Parse --icicle flag
for arg in "$@"; do
    if [ "$arg" = "--icicle" ]; then
        FEATURES="host,icicle"
        ICICLE_DIR="$(pwd)/target/release/deps/icicle"
        ICICLE_ENV="ICICLE_BACKEND_INSTALL_DIR=$ICICLE_DIR DYLD_LIBRARY_PATH=$ICICLE_DIR/lib"
        export ICICLE_BACKEND_INSTALL_DIR="$ICICLE_DIR"
        export DYLD_LIBRARY_PATH="$ICICLE_DIR/lib"
    fi
done

PGO_DIR="/tmp/pgo-data"
MERGED="/tmp/pgo-merged.profdata"
LLVM_PROFDATA="${LLVM_PROFDATA:-/Library/Developer/CommandLineTools/usr/bin/llvm-profdata}"

echo "=== PGO Build for Jolt Prover ==="
echo "Benchmark: $BENCH"
echo "Features: $FEATURES"
echo ""

# Step 1: Clean build to cache guest binary
echo "[1/5] Building normally to cache guest..."
cargo build --release -p jolt-core --features "$FEATURES" -q
./target/release/jolt-core profile --name "$BENCH" --format chrome >/dev/null 2>&1
echo "  Guest cached."

# Step 2: Instrumented build
echo "[2/5] Building with PGO instrumentation..."
rm -rf "$PGO_DIR" && mkdir -p "$PGO_DIR"
RUSTFLAGS="-Cprofile-generate=$PGO_DIR" cargo build --release -p jolt-core --features "$FEATURES" -q
echo "  Instrumented build done."

# Step 3: Collect profile data
echo "[3/5] Collecting profile data..."
./target/release/jolt-core profile --name "$BENCH" --format chrome >/dev/null 2>&1
PROFRAW_COUNT=$(ls "$PGO_DIR"/*.profraw 2>/dev/null | wc -l)
echo "  Collected $PROFRAW_COUNT profile(s)."

# Step 4: Merge profiles
echo "[4/5] Merging profiles..."
"$LLVM_PROFDATA" merge -o "$MERGED" "$PGO_DIR/"
echo "  Merged to $MERGED ($(du -h "$MERGED" | cut -f1))."

# Step 5: PGO-optimized build
echo "[5/5] Building with PGO optimization..."
RUSTFLAGS="-Cprofile-use=$MERGED -Cllvm-args=-pgo-warn-missing-function=false" cargo build --release -p jolt-core --features "$FEATURES" -q
echo "  PGO-optimized build done."

echo ""
echo "=== PGO build complete ==="
echo "Binary: ./target/release/jolt-core"
echo "Run: ./target/release/jolt-core profile --name $BENCH --format chrome"
if [ -n "$ICICLE_ENV" ]; then
    echo "Note: Set $ICICLE_ENV before running"
fi
echo ""
echo "To run tests with PGO:"
echo "  RUSTFLAGS=\"-Cprofile-use=$MERGED -Cllvm-args=-pgo-warn-missing-function=false\" cargo nextest run -p jolt-core muldiv --cargo-quiet --features $FEATURES"
