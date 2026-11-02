#!/usr/bin/env bash
set -euo pipefail

CASE_NAME="${1:?usage: run_nsys_trace.sh CASE_NAME OUTPUT_DIR}"
OUTPUT_DIR="${2:?usage: run_nsys_trace.sh CASE_NAME OUTPUT_DIR}"
MODE="${MODE:-qr}"
METHODS="${METHODS:-all}"
WARMUPS="${WARMUPS:-2}"
TRIALS="${TRIALS:-2}"
MAX_REFINEMENTS="${MAX_REFINEMENTS:-5}"
TOLERANCE="${TOLERANCE:-1e-12}"
SEED="${SEED:-1}"
NS_INVERSE_ITERATIONS="${NS_INVERSE_ITERATIONS:-20}"
NS_POLAR_ITERATIONS="${NS_POLAR_ITERATIONS:-20}"
NS_INVERSE_TOLERANCE="${NS_INVERSE_TOLERANCE:-1e-10}"

if [ -z "${NSYS:-}" ]; then
    if command -v nsys >/dev/null 2>&1; then
        NSYS="$(command -v nsys)"
    else
        NSYS="$(find /soft/nvidia /usr/local/cuda /opt/nvidia -path '*/bin/nsys' -type f 2>/dev/null | sort -V | tail -n 1 || true)"
    fi
fi
if [ -z "${NSYS:-}" ] || [ ! -x "$NSYS" ]; then
    echo "ERROR: nsys not found" >&2
    exit 20
fi

case "$CASE_NAME" in
    moderate)
        M=4096; N=256; CONDITION=100.0
        ;;
    high)
        M=20000; N=2000; CONDITION=10000000000.0
        ;;
    ns)
        M=4096; N=256; CONDITION=1.0; MODE=q
        ;;
    *)
        echo "ERROR: unknown case '$CASE_NAME'; expected moderate, high, or ns" >&2
        exit 30
        ;;
esac

mkdir -p "$OUTPUT_DIR"
SAFE_CASE="${CASE_NAME}_m${M}_n${N}_${MODE}"
REP="${OUTPUT_DIR%/}/${SAFE_CASE}"
LOG="${REP}.log"
CSV="${REP}.csv"

echo "nsys=$NSYS" | tee "$LOG"
echo "case=$CASE_NAME m=$M n=$N condition=$CONDITION output=$MODE methods=$METHODS warmups=$WARMUPS trials=$TRIALS" | tee -a "$LOG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}" | tee -a "$LOG"
nvidia-smi -L 2>&1 | tee -a "$LOG" || true

"$NSYS" profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output="$REP" \
    ./qr8_benchmark \
        --m "$M" --n "$N" --condition "$CONDITION" --seed "$SEED" \
        --warmups "$WARMUPS" --trials "$TRIALS" \
        --maximum-refinements "$MAX_REFINEMENTS" --tolerance "$TOLERANCE" \
        --output "$MODE" \
        --methods "$METHODS" \
        --ns-inverse-iterations "$NS_INVERSE_ITERATIONS" \
        --ns-polar-iterations "$NS_POLAR_ITERATIONS" \
        --ns-inverse-tolerance "$NS_INVERSE_TOLERANCE" \
        --matrix-id "$SAFE_CASE" \
        --raw-csv "$CSV" \
        2>&1 | tee -a "$LOG"

"$NSYS" stats \
    --report cuda_api_sum \
    --report cuda_gpu_kern_sum \
    --report cuda_gpu_mem_time_sum \
    --format csv \
    --output "$REP" \
    "$REP.nsys-rep" \
    2>&1 | tee -a "$LOG" || true

python3 tests/validate_results.py "$CSV" | tee -a "$LOG"
echo "trace=$REP.nsys-rep" | tee -a "$LOG"
