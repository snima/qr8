#!/bin/bash
#SBATCH --job-name=qr8_v2_h200
#SBATCH --partition=P2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H200:2
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

JOB_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/runs/h200_v2_${JOB_ID}}"
MODE="${MODE:-validation}"
CUDA_ARCH="${CUDA_ARCH:-sm_90}"
WARMUPS="${WARMUPS:-5}"
TRIALS="${TRIALS:-30}"
MAX_REFINEMENTS="${MAX_REFINEMENTS:-5}"
TOLERANCE="${TOLERANCE:-1e-12}"
OUTPUT="${OUTPUT:-both}"
EQUILIBRATE="${EQUILIBRATE:-0}"
INCLUDE_RANK_DEFICIENT="${INCLUDE_RANK_DEFICIENT:-0}"
NS_INVERSE_ITERATIONS="${NS_INVERSE_ITERATIONS:-20}"
NS_POLAR_ITERATIONS="${NS_POLAR_ITERATIONS:-20}"
NS_INVERSE_TOLERANCE="${NS_INVERSE_TOLERANCE:-1e-10}"
L2_SHIFT="${L2_SHIFT:-0.0}"
L2_SHIFT_RETRIES="${L2_SHIFT_RETRIES:-0}"
OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p "$RUN_ROOT" results
cp "$0" "$RUN_ROOT/submit_h200_v2.sbatch" 2>/dev/null || true

{
    echo "job_id=$JOB_ID"
    echo "mode=$MODE"
    echo "project_dir=$PROJECT_DIR"
    echo "run_root=$RUN_ROOT"
    echo "cuda_arch=$CUDA_ARCH"
    echo "warmups=$WARMUPS"
    echo "trials=$TRIALS"
    echo "max_refinements=$MAX_REFINEMENTS"
    echo "tolerance=$TOLERANCE"
    echo "output=$OUTPUT"
    echo "equilibrate=$EQUILIBRATE"
    echo "include_rank_deficient=$INCLUDE_RANK_DEFICIENT"
    echo "date=$(date)"
    echo "hostname=$(hostname)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
    echo "l2_shift=$L2_SHIFT"
    echo "l2_shift_retries=$L2_SHIFT_RETRIES"
} | tee "$RUN_ROOT/job_config.txt"

source /etc/profile.d/modules.sh 2>/dev/null || true
source /usr/share/Modules/init/bash 2>/dev/null || true
if type module >/dev/null 2>&1; then
    module list 2>&1 || true
    module load cuda-12.1.module 2>/dev/null || \
    module load cuda/12.1 2>/dev/null || \
    module load cuda-12.1 2>/dev/null || \
    module load cuda-12.2 2>/dev/null || \
    module load cuda-12.3 2>/dev/null || \
    module load cuda-12.4 2>/dev/null || true
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not available after CUDA module setup" | tee -a "$RUN_ROOT/job_config.txt"
    exit 20
fi

NVCC_REL="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p' | head -n 1 || true)"
read -r NVCC_MAJ NVCC_MIN <<< "${NVCC_REL:-0 0}"
if [ "${NVCC_MAJ:-0}" -lt 11 ] || { [ "${NVCC_MAJ:-0}" -eq 11 ] && [ "${NVCC_MIN:-0}" -lt 8 ]; }; then
    echo "ERROR: CUDA ${NVCC_MAJ:-0}.${NVCC_MIN:-0} is too old for H200/sm_90" | tee -a "$RUN_ROOT/job_config.txt"
    exit 21
fi

{
    echo
    echo "=== compiler ==="
    command -v nvcc
    nvcc --version
    echo
    echo "=== nvidia-smi ==="
    nvidia-smi
    echo
    echo "=== topology ==="
    nvidia-smi topo -m || true
    echo
    echo "=== gpu query ==="
    nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,compute_cap,driver_version,temperature.gpu,power.draw,power.limit,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv || true
} 2>&1 | tee "$RUN_ROOT/environment_before_build.txt"

make clean
make CUDA_ARCH="$CUDA_ARCH"

make smoke
python3 tests/validate_results.py results/smoke.csv

if [ "$MODE" = "validation" ]; then
    ./qr8_benchmark --m 513 --n 31 --warmups 1 --trials 2 --maximum-refinements 3 \
        --raw-csv "$RUN_ROOT/odd_synthetic_validation.csv"
    python3 tests/validate_results.py "$RUN_ROOT/odd_synthetic_validation.csv"

    ./qr8_benchmark --m 4096 --n 256 --condition 100 --output q --warmups 1 --trials 3 \
        --ns-inverse-iterations "$NS_INVERSE_ITERATIONS" \
        --ns-polar-iterations "$NS_POLAR_ITERATIONS" \
        --ns-inverse-tolerance "$NS_INVERSE_TOLERANCE" \
        --raw-csv "$RUN_ROOT/ns_validation.csv"
    python3 tests/validate_results.py "$RUN_ROOT/ns_validation.csv"

    make suite-smoke
    LATEST_RAW="$(find results/suite-smoke -name raw_all.csv -type f | sort | tail -n 1)"
    python3 tests/validate_results.py "$LATEST_RAW"
elif [ "$MODE" = "full" ]; then
    RUN_ARGS=(--warmups "$WARMUPS" --trials "$TRIALS" --maximum-refinements "$MAX_REFINEMENTS" --tolerance "$TOLERANCE" --output "$OUTPUT" --results "$RUN_ROOT/primary")
    if [ "$EQUILIBRATE" = "1" ]; then
        RUN_ARGS+=(--equilibrate)
    fi
    if [ "$INCLUDE_RANK_DEFICIENT" = "1" ]; then
        RUN_ARGS+=(--include-rank-deficient)
    fi
    python3 scripts/run_suite.py "${RUN_ARGS[@]}"
elif [ "$MODE" = "synthetic" ]; then
    SYNTH_GRID="${SYNTHETIC_GRID:-config/synthetic_grid.json}"
    python3 scripts/run_suite.py --synthetic-only --synthetic-grid "$SYNTH_GRID" \
        --warmups "$WARMUPS" --trials "$TRIALS" \
        --maximum-refinements "$MAX_REFINEMENTS" --tolerance "$TOLERANCE" --output "$OUTPUT" \
        --l2-shift "$L2_SHIFT" --l2-shift-retries "$L2_SHIFT_RETRIES" \
        --results "$RUN_ROOT/synthetic"
elif [ "$MODE" = "real" ]; then
    REAL_ARGS=(--real-only --warmups "$WARMUPS" --trials "$TRIALS" --maximum-refinements "$MAX_REFINEMENTS" --tolerance "$TOLERANCE" --output "$OUTPUT" --results "$RUN_ROOT/real")
    if [ "$INCLUDE_RANK_DEFICIENT" = "1" ]; then
        REAL_ARGS+=(--include-rank-deficient)
    fi
    python3 scripts/run_suite.py "${REAL_ARGS[@]}"
elif [ "$MODE" = "instrumented" ]; then
    python3 scripts/run_suite.py --kernel-timing --warmups "$WARMUPS" --trials "$TRIALS" \
        --maximum-refinements "$MAX_REFINEMENTS" --tolerance "$TOLERANCE" --output qr \
        --results "$RUN_ROOT/instrumented"
else
    echo "ERROR: unknown MODE=$MODE" | tee -a "$RUN_ROOT/job_config.txt"
    exit 30
fi

PACKAGE_DIR="${PACKAGE_DIR:-$PROJECT_DIR/packages}"
mkdir -p "$PACKAGE_DIR"
PACK="$PACKAGE_DIR/qr8_v2_h200_${JOB_ID}.tar.gz"
tar -C "$RUN_ROOT" -czf "$PACK" .

echo
echo "============================================================"
echo "job completed: $(date)"
echo "run root: $RUN_ROOT"
echo "package: $PACK"
echo "============================================================"
