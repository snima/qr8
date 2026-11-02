# QR8: Mixed-Precision Cholesky-QR with INT8 Initialization and FP64 Refinement

[![SIAM SISC](https://img.shields.io/badge/SIAM%20Journal%20on%20Scientific%20Computing-Submitted-005587.svg)](https://www.siam.org/journals/sisc)
[![Reproducibility Badge](https://img.shields.io/badge/SIAM%20SISC%20Reproducibility-Code%20%26%20Data%20Available-green.svg)](https://sisc.siam.org/html/sisc_reproducibility.html)
[![License: Beerware](https://img.shields.io/badge/License-Beerware%20Rev%2042-orange.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B%20%7C%2012.x-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/Standard-C%2B%2B17-00599C.svg?logo=c%2B%2B)](https://en.cppreference.com/w/cpp/17)
[![Project Website](https://img.shields.io/badge/Project%20Page-snima.github.io%2Fqr8-blueviolet.svg)](https://snima.github.io/qr8/)

This repository provides the official implementation, CUDA benchmark harness, experimental matrices, and reproducibility materials for the paper:

> **Mixed-Precision Cholesky-QR with INT8 Initialization and FP64 Refinement for Very Tall Matrices on Heterogeneous GPUs**  
> *Nima Sahraneshinsamani, José I. Aliaga, Sandra Catalán, and José R. Herrero*  
> Universitat Jaume I & Universitat Politècnica de Catalunya  
> Submitted to **SIAM Journal on Scientific Computing (SISC)**, Special Section on Iterative and Multigrid Methods (2026).

---

## Overview

Conventional Cholesky-QR algorithms achieve peak BLAS-3 throughput for tall-and-skinny matrices ($m \gg n$) on GPUs by reducing the dominant work to matrix multiplication ($G = A^TA$) and triangular solves. However, they suffer from quadratic condition number growth ($\kappa(G) = \kappa(A)^2$) and breakdown on ill-conditioned inputs.

**QR8** introduces a mixed-precision orthogonalization pipeline that:
1. **Forms a coarse initial Gram matrix in INT8/INT32** using high-throughput Tensor Cores or `DP4A` vector instructions (`cublasGemmEx`), where exact 32-bit integer accumulation completely eliminates floating-point accumulation drift ($\Delta G_{\mathrm{acc}} = 0$) and guarantees positive semi-definiteness ($B_L^T B_L \succeq 0$).
2. **Employs two INT8 initializers:** single-slice (`INT8-L1`) and two-slice residual-corrected (`INT8-L2`), where L2 quantization reduces the initial residual scale by a factor of $2q = 254$ ($\approx 48$ dB), expanding the admissible condition number range for single-pass refinement by roughly two orders of magnitude.
3. **Applies adaptive FP64 Cholesky-QR refinement with residual-variance diagonal shifting:** $s_{\mathrm{var}} = \frac{m}{12} \sum_{j=1}^n \alpha_{2j}^2$ guarantees 100% completion up to $\kappa(A) = 10^{10}$ (where standard unshifted `FP64-CholQR2` fails).
4. **Delivers real GPU speed-ups:** up to $1.28\times$ validated speed-up over matched adaptive FP64 Cholesky-QR, $1.1\times$--$5.1\times$ over vendor `cuSOLVER-Householder-QR`, and up to $10.0\times$ when an unrefined coarse basis suffices.

---

## Summary of Implemented Methods

| Method | Precision / Formulation | Description |
|---|---|---|
| `INT8L1-CholeskyQR-Adaptive` | INT8/INT32 + FP64 Refinement | Single-product INT8 initialization with adaptive FP64 refinement |
| `INT8L2-CholeskyQR-Adaptive` | INT8/INT32 + FP64 Refinement | Three-product residual-corrected INT8 initializer + adaptive FP64 path |
| `INT8L1-CholeskyQR` | Pure INT8 Initialization + 1 solve | Coarse unrefined orthogonal basis |
| `INT8L2-CholeskyQR` | Pure INT8 (Residual) + 1 solve | Coarse unrefined orthogonal basis with residual correction |
| `FP64-CholeskyQR2` | FP64 | Standard two-pass Cholesky-QR (Fukaya et al., 2014) |
| `FP64-shiftedCholeskyQR` | FP64 | Shifted Cholesky-QR (Alg. 2.1, Fukaya et al., 2020) |
| `FP64-shiftedCholeskyQR3` | FP64 | Shifted Cholesky-QR3 (Alg. 4.2, Fukaya et al., 2020) |
| `FP64-CholeskyQR-Adaptive-Control` | FP64 | Matched adaptive control baseline |
| `cuSOLVER-Householder-QR` | FP64 | Vendor Householder QR (`Dgeqrf` + `Dorgqr`) |

---

## Quickstart

### Prerequisites
- NVIDIA GPU with Compute Capability $\ge 7.0$ (e.g., Volta V100, Turing, Ampere RTX A6000, Hopper H100/H200).
- CUDA Toolkit 11.8 or 12.x (`nvcc`, `cuBLAS`, `cuSOLVER`, `cuRAND`).
- C++17 compatible compiler (GCC $\ge 8$, Clang $\ge 7$).
- Python 3.8+ (for validation and result analysis scripts).

### 1. Build
```bash
git clone https://github.com/snima/qr8.git
cd qr8

# Build for your GPU architecture (native detection):
make

# Or specify your target architecture:
make CUDA_ARCH=sm_70   # NVIDIA Tesla V100 / V100S
make CUDA_ARCH=sm_86   # NVIDIA RTX A6000 / RTX 3090
make CUDA_ARCH=sm_90   # NVIDIA Hopper H100 / H200 NVL
```

### 2. Run Smoke Test
The smoke test compiles the harness, executes synthetic trials across all methods, logs CUDA-event metrics, and validates numerical invariants:
```bash
make smoke
```
Expected output:
```text
./qr8_benchmark --m 512 --n 32 --warmups 1 --trials 2 --maximum-refinements 2 --raw-csv results/smoke.csv
FP64-CholeskyQR2: successes=2/2 ...
...
INT8L1-CholeskyQR-Adaptive: successes=2/2 ...
INT8L2-CholeskyQR-Adaptive: successes=2/2 ...
python3 tests/validate_results.py results/smoke.csv
validated 30 rows from results/smoke.csv
```

---

## Reproducing Paper Experiments

### Synthetic Conditioning Grid Sweep
Run the paired multi-trial benchmark across condition numbers $\kappa(A) \in [10^1, 10^{12}]$:
```bash
python3 scripts/run_suite.py \
    --synthetic-only \
    --synthetic-grid config/synthetic_grid.json \
    --warmups 5 \
    --trials 90 \
    --output both \
    --results results/synthetic_suite
```

### SuiteSparse Application Matrices
Run the evaluation on real-world sparse/least-squares benchmark problems located in `data/`:
```bash
python3 scripts/run_suite.py \
    --datasets-only \
    --datasets-config config/datasets.json \
    --warmups 5 \
    --trials 30 \
    --results results/suitesparse_suite
```

### Slurm Cluster Batch Scripts
Cluster job scripts for NVIDIA V100S, RTX A6000, and H200 NVL are located in `scripts/`:
```bash
sbatch scripts/submit_v100s_v2.sbatch
sbatch scripts/submit_a6000_v2.sbatch
sbatch scripts/submit_h200_v2.sbatch
```

---

## Repository Structure

```text
qr8/
├── Makefile                # Build system (sm_70, sm_86, sm_90, smoke tests)
├── README.md               # Main repository documentation
├── LICENSE                 # Beerware License (Revision 42)
├── src/
│   └── qr8_benchmark.cu    # Complete CUDA benchmark implementation
├── include/
│   └── benchmark_types.hpp # Algorithm enums, options, and data structures
├── config/                 # Benchmark grids (synthetic, dense sweep, datasets)
├── data/                   # SuiteSparse matrix inputs (.qr8, .json)
├── scripts/                # Benchmark driver and analysis scripts
├── tests/                  # Numerical invariance and output validation
└── docs/                   # GitHub Pages interactive project website
    └── index.html          # Interactive research page & conditioning explorer
```

---

## License

Released under the **QR8 Beerware License (Revision 42)**.  
See [LICENSE](LICENSE) for details.
