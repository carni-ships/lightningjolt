#!/bin/bash
# run_full_benchmark.sh - Run full prover benchmark and save results
#
# Usage:
#   ./run_full_benchmark.sh <RPC_URL> <CACHE_DIR> <BLOCK_NUMBER> [NUM_TX]
#
# Output:
#   - benchmarks/results/YYYYMMDD_HHMMSS_benchmark.json (JSON results)
#   - benchmarks/results/YYYYMMDD_HHMMSS_benchmark.log (full log)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$RESULTS_DIR/${TIMESTAMP}_benchmark.log"
JSON_FILE="$RESULTS_DIR/${TIMESTAMP}_benchmark.json"

# Default values
RPC_URL="${1:-}"
CACHE_DIR="${2:-/tmp/ethrex-block-cache}"
BLOCK_NUM="${3:-24922263}"
NUM_TX="${4:-10}"

# Validate inputs
if [ -z "$RPC_URL" ]; then
    echo "Error: RPC URL required"
    echo "Usage: $0 <RPC_URL> <CACHE_DIR> <BLOCK_NUMBER> [NUM_TX]"
    exit 1
fi

echo "============================================================"
echo "PROVER PHASE BREAKDOWN BENCHMARK"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo "Block: #$BLOCK_NUM"
echo "Transactions: $NUM_TX"
echo "Log file: $LOG_FILE"
echo "============================================================"
echo ""

# Run the benchmark and capture output
START_TIME=$(date +%s)

{
    echo "============================================================"
    echo "PROVER BENCHMARK - $TIMESTAMP"
    echo "============================================================"
    echo "Block: #$BLOCK_NUM"
    echo "RPC: $RPC_URL"
    echo "Transactions: $NUM_TX"
    echo "============================================================"
    echo ""

    timeout 1800 cargo run -p ethrex-trace --example ethrex_realtime_prove -- \
        "$RPC_URL" "$CACHE_DIR" "$BLOCK_NUM"

} 2>&1 | tee "$LOG_FILE"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "Benchmark completed in ${ELAPSED}s"
echo "Log saved to: $LOG_FILE"
echo "============================================================"
echo ""

# Extract phase timings and create JSON
echo "Extracting phase timings..."

# Parse log file and create JSON summary
python3 - <<PYTHON_SCRIPT
import re
import json
import sys
from datetime import datetime

log_file = "$LOG_FILE"
json_file = "$JSON_FILE"

phases = []
total_time_ms = 0
tx_count = 0
steps_count = 0
block_number = "$BLOCK_NUM"

with open(log_file, 'r') as f:
    content = f.read()
    for line in content.split('\n'):
        # Parse [PROFILE] lines
        match = re.search(r'\[PROFILE\]\s+([^:]+):\s+(\d+)ms', line)
        if match:
            phase = match.group(1).strip()
            time_ms = int(match.group(2))
            phases.append({
                "phase": phase,
                "time_ms": time_ms,
                "time_s": round(time_ms / 1000, 2)
            })
            total_time_ms += time_ms

        # Extract metrics
        tx_match = re.search(r'Transactions proved:\s+(\d+)', line)
        if tx_match:
            tx_count = int(tx_match.group(1))

        steps_match = re.search(r'(\d+)\s+steps', line)
        if steps_match:
            steps_count = int(steps_match.group(1))

# Add percentages
for p in phases:
    p["percentage"] = round((p["time_ms"] / total_time_ms) * 100, 1) if total_time_ms > 0 else 0

# Create result
result = {
    "benchmark": "prover_phase_breakdown",
    "timestamp": "$TIMESTAMP",
    "block_number": int(block_number),
    "config": {
        "rpc_url": "$RPC_URL",
        "cache_dir": "$CACHE_DIR",
        "num_tx": $NUM_TX
    },
    "metrics": {
        "tx_count": tx_count,
        "steps_count": steps_count,
        "total_time_ms": total_time_ms,
        "total_time_s": round(total_time_ms / 1000, 2),
        "throughput_tx_per_sec": round(tx_count / (total_time_ms / 1000), 3) if total_time_ms > 0 and tx_count > 0 else 0
    },
    "phases": phases,
    "log_file": log_file
}

with open(json_file, 'w') as f:
    json.dump(result, f, indent=2)

print(f"JSON results saved to: {json_file}")
PYTHON_SCRIPT

echo ""
echo "Results saved to:"
echo "  - Log: $LOG_FILE"
echo "  - JSON: $JSON_FILE"
echo ""

# Print summary
echo "PHASE BREAKDOWN:"
echo "============================================================"
python3 -c "
import json
with open('$JSON_FILE') as f:
    data = json.load(f)
    print(f\"{'Phase':<40} {'Time (ms)':<12} {'%':<8}\")
    print('-' * 60)
    for p in data['phases']:
        print(f\"{p['phase']:<40} {p['time_ms']:<12,} {p['percentage']:>6.1f}%\")
    print('-' * 60)
    print(f\"{'TOTAL':<40} {data['metrics']['total_time_ms']:<12,} {'100.0':>6}%\")
    print()
    print(f\"Transactions: {data['metrics']['tx_count']}\")
    print(f\"Steps: {data['metrics']['steps_count']}\")
    print(f\"Throughput: {data['metrics']['throughput_tx_per_sec']:.3f} tx/s\")
"
