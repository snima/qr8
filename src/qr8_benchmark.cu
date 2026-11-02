#include "benchmark_types.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { const cudaError_t status_ = (call); if (status_ != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status_)); } while (0)
#define CUBLAS_CHECK(call) do { const cublasStatus_t status_ = (call); if (status_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(static_cast<int>(status_))); } while (0)
#define CUSOLVER_CHECK(call) do { const cusolverStatus_t status_ = (call); if (status_ != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER status " + std::to_string(static_cast<int>(status_))); } while (0)
#define CURAND_CHECK(call) do { const curandStatus_t status_ = (call); if (status_ != CURAND_STATUS_SUCCESS) throw std::runtime_error("cuRAND status " + std::to_string(static_cast<int>(status_))); } while (0)
#define KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

namespace qr8 {

const char* algorithm_name(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::CholeskyQR2: return "FP64-CholeskyQR2";
        case Algorithm::ShiftedCholeskyQR: return "FP64-shiftedCholeskyQR";
        case Algorithm::ShiftedCholeskyQR3: return "FP64-shiftedCholeskyQR3";
        case Algorithm::AdaptiveFP64: return "FP64-CholeskyQR-Adaptive-Control";
        case Algorithm::Int8L1Adaptive: return "INT8L1-CholeskyQR-Adaptive";
        case Algorithm::Int8L2Adaptive: return "INT8L2-CholeskyQR-Adaptive";
        case Algorithm::Int8L1CholeskyQR: return "INT8L1-CholeskyQR";
        case Algorithm::Int8L2CholeskyQR: return "INT8L2-CholeskyQR";
        case Algorithm::Int8L2PolarNS: return "INT8L2-PolarNS-Q";
        case Algorithm::Int8L2PolarNSQR: return "INT8L2-PolarNS-QR";
        case Algorithm::Int8L2PolarChebyshev: return "INT8L2-PolarChebyshev-Q";
        case Algorithm::Int8L2PolarChebyshevQR: return "INT8L2-PolarChebyshev-QR";
        case Algorithm::Int8L2NIMARefinement: return "INT8L2-NIMA-Refinement-Q";
        case Algorithm::Int8L2NIMARefinementQR: return "INT8L2-NIMA-Refinement-QR";
        case Algorithm::Int8L2ShiftRetry: return "INT8L2-ShiftRetry-Q";
        case Algorithm::Int8L2ShiftRetryQR: return "INT8L2-ShiftRetry-QR";
        case Algorithm::HouseholderQR: return "cuSOLVER-Householder-QR";
        case Algorithm::Int8L2CertifiedRidge: return "INT8L2-CertifiedRidge-Q";
        case Algorithm::Int8L2CertifiedRidgeQR: return "INT8L2-CertifiedRidge-QR";
        case Algorithm::Int8L2TriangularPolish: return "INT8L2-TriangularPolish-Q";
        case Algorithm::Int8L2TriangularPolishQR: return "INT8L2-TriangularPolish-QR";
    }
    return "unknown";
}

const char* output_name(OutputContract output) { return output == OutputContract::QOnly ? "Q" : "QR"; }

const char* stop_reason_name(StopReason reason) {
    switch (reason) {
        case StopReason::FixedPasses: return "fixed_passes";
        case StopReason::ToleranceMet: return "tolerance_met";
        case StopReason::PassLimit: return "pass_limit";
        case StopReason::PotrfFailure: return "potrf_failure";
        case StopReason::InverseSqrtLimit: return "inverse_sqrt_limit";
        case StopReason::PolarLimit: return "polar_limit";
        case StopReason::ChebyshevDegreeLimit: return "chebyshev_degree_limit";
        case StopReason::InvalidInput: return "invalid_input";
    }
    return "unknown";
}

TrialStatistics summarize_times(const std::vector<RunResult>& results) {
    std::vector<double> values;
    for (const RunResult& result : results) if (result.factorization_success) values.push_back(result.time.total_ms);
    TrialStatistics stats;
    stats.count = values.size();
    if (values.empty()) return stats;
    stats.mean_ms = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double squares = 0.0;
    for (double value : values) squares += (value - stats.mean_ms) * (value - stats.mean_ms);
    stats.standard_deviation_ms = values.size() > 1 ? std::sqrt(squares / (values.size() - 1)) : 0.0;
    stats.minimum_ms = *std::min_element(values.begin(), values.end());
    std::sort(values.begin(), values.end());
    stats.median_ms = values.size() % 2 ? values[values.size() / 2]
                                      : 0.5 * (values[values.size() / 2 - 1] + values[values.size() / 2]);
    return stats;
}

template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() { if (pointer_) cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : pointer_(other.pointer_), count_(other.count_) {
        other.pointer_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this == &other) return *this;
        if (pointer_) cudaFree(pointer_);
        pointer_ = other.pointer_;
        count_ = other.count_;
        other.pointer_ = nullptr;
        other.count_ = 0;
        return *this;
    }
    void allocate(std::size_t count) {
        T* replacement = nullptr;
        if (count > 0) CUDA_CHECK(cudaMalloc(&replacement, count * sizeof(T)));
        if (pointer_) {
            const cudaError_t status = cudaFree(pointer_);
            if (status != cudaSuccess) {
                if (replacement) cudaFree(replacement);
                throw std::runtime_error(cudaGetErrorString(status));
            }
        }
        pointer_ = replacement;
        count_ = count;
    }
    T* get() { return pointer_; }
    const T* get() const { return pointer_; }
    std::size_t size() const { return count_; }
private:
    T* pointer_ = nullptr;
    std::size_t count_ = 0;
};

class LibraryContext {
public:
    LibraryContext() {
        try {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            CUBLAS_CHECK(cublasCreate(&blas));
            CUSOLVER_CHECK(cusolverDnCreate(&solver));
            CUBLAS_CHECK(cublasSetStream(blas, stream));
            CUSOLVER_CHECK(cusolverDnSetStream(solver, stream));
            CUBLAS_CHECK(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
        } catch (...) {
            if (solver) cusolverDnDestroy(solver);
            if (blas) cublasDestroy(blas);
            if (stream) cudaStreamDestroy(stream);
            throw;
        }
    }
    ~LibraryContext() {
        if (solver) cusolverDnDestroy(solver);
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
    }
    LibraryContext(const LibraryContext&) = delete;
    LibraryContext& operator=(const LibraryContext&) = delete;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
};

__global__ void column_norms(const double* matrix, double* norms, int m, int n) {
    const int column = blockIdx.x;
    if (column >= n) return;
    extern __shared__ double values[];
    double sum = 0.0;
    for (int row = threadIdx.x; row < m; row += blockDim.x) {
        const double value = matrix[static_cast<std::size_t>(column) * m + row];
        sum += value * value;
    }
    values[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset) values[threadIdx.x] += values[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) norms[column] = std::sqrt(values[0]);
}

__global__ void divide_columns(double* matrix, const double* norms, int m, int n) {
    const int column = blockIdx.x;
    if (column >= n) return;
    const double norm = norms[column];
    const double inverse = norm > 0.0 ? 1.0 / norm : 0.0;
    for (int row = threadIdx.x; row < m; row += blockDim.x)
        matrix[static_cast<std::size_t>(column) * m + row] *= inverse;
}

__global__ void scale_columns(double* matrix, const double* scales, int m, int n) {
    const int column = blockIdx.x;
    const int row = blockIdx.y * blockDim.x + threadIdx.x;
    if (column < n && row < m)
        matrix[static_cast<std::size_t>(column) * m + row] *= scales[column];
}

__global__ void multiply_r_columns(double* matrix, const double* scales, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && column < n) matrix[static_cast<std::size_t>(column) * n + row] *= scales[column];
}

__global__ void extract_upper(const double* source, int leading_dimension, double* destination, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && column < n)
        destination[static_cast<std::size_t>(column) * n + row] = row <= column ? source[static_cast<std::size_t>(column) * leading_dimension + row] : 0.0;
}

__global__ void add_diagonal(double* matrix, int n, double shift) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) matrix[static_cast<std::size_t>(index) * n + index] += shift;
}

__global__ void form_gram_minus_identity(double* full, const double* upper, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || column >= n) return;
    double value = row <= column ? upper[static_cast<std::size_t>(column) * n + row]
                                 : upper[static_cast<std::size_t>(row) * n + column];
    if (row == column) value -= 1.0;
    full[static_cast<std::size_t>(column) * n + row] = value;
}

__global__ void set_identity(double* matrix, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && column < n)
        matrix[static_cast<std::size_t>(column) * n + row] = row == column ? 1.0 : 0.0;
}

__global__ void subtract_identity_full(double* matrix, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) matrix[static_cast<std::size_t>(index) * n + index] -= 1.0;
}

// Converts E = ZY-I in place to I-E/2 = (3I-ZY)/2.
__global__ void inverse_sqrt_transform_from_residual(double* matrix, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && column < n) {
        double value = -0.5 * matrix[static_cast<std::size_t>(column) * n + row];
        if (row == column) value += 1.0;
        matrix[static_cast<std::size_t>(column) * n + row] = value;
    }
}

// Converts E = Q^TQ-I in place to I-E/2 = (3I-Q^TQ)/2.
__global__ void polar_transform_from_residual(double* matrix, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && column < n) {
        double value = -0.5 * matrix[static_cast<std::size_t>(column) * n + row];
        if (row == column) value += 1.0;
        matrix[static_cast<std::size_t>(column) * n + row] = value;
    }
}

// Theorem 4.7: Decomposes E = Q^T Q - I into T = triu(E, 1) + 0.5 * diag(E)
// and forms P = I - T in destination buffer P.
// Inputs: upper triangle of Q^T Q stored in `gram` (column-major n x n)
// Output: P = I - T in `P` (strictly upper triangular, column-major n x n)
__global__ void form_triangular_polish_matrix(const double* __restrict__ gram,
                                              double* __restrict__ P,
                                              int n) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && column < n) {
        const std::size_t idx = static_cast<std::size_t>(column) * n + row;
        if (row < column) {
            // Off-diagonal: E_ij = gram_ij, T_ij = E_ij, P_ij = -T_ij = -gram_ij
            P[idx] = -gram[idx];
        } else if (row == column) {
            // Diagonal: E_ii = gram_ii - 1.0, T_ii = 0.5 * E_ii, P_ii = 1.0 - 0.5 * E_ii = 1.5 - 0.5 * gram_ii
            P[idx] = 1.5 - 0.5 * gram[idx];
        } else {
            // Strictly lower triangular: 0.0
            P[idx] = 0.0;
        }
    }
}

__global__ void symmetrize_upper_to_full(double* matrix, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && column < n && row > column)
        matrix[static_cast<std::size_t>(column) * n + row] =
            matrix[static_cast<std::size_t>(row) * n + column];
}

__device__ __forceinline__ double warp_reduce_max(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ double block_reduce_max(double val, double* smem) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.0;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}

__global__ void quantize_columns(const double* matrix, std::int8_t* quantized, double* scales, int m, int n, int ldq) {
    const int column = blockIdx.x;
    if (column >= n) return;
    __shared__ double smem[32];
    double maximum = 0.0;
    for (int row = threadIdx.x; row < m; row += blockDim.x)
        maximum = fmax(maximum, fabs(matrix[static_cast<std::size_t>(column) * m + row]));
    double max_val = block_reduce_max(maximum, smem);
    if (threadIdx.x == 0) smem[0] = max_val;
    __syncthreads();
    max_val = smem[0];
    const double scale = max_val > 0.0 ? max_val / 127.0 : 1.0;
    const double inverse = max_val > 0.0 ? 127.0 / max_val : 0.0;
    if (threadIdx.x == 0) scales[column] = scale;
    for (int row = threadIdx.x; row < m; row += blockDim.x) {
        double value = nearbyint(matrix[static_cast<std::size_t>(column) * m + row] * inverse);
        value = fmax(-127.0, fmin(127.0, value));
        quantized[static_cast<std::size_t>(column) * ldq + row] = static_cast<std::int8_t>(value);
    }
}

// Fused Two-Slice Quantization: computes X1, scale1, and X2, scale2 directly in registers
// eliminating intermediate DRAM writes and reads of FP64 residual matrix.
__global__ void fused_two_slice_quantize(const double* __restrict__ matrix,
                                         std::int8_t* __restrict__ X1,
                                         double* __restrict__ scale1,
                                         std::int8_t* __restrict__ X2,
                                         double* __restrict__ scale2,
                                         int m, int n, int ldq) {
    const int column = blockIdx.x;
    if (column >= n) return;
    __shared__ double smem[32];
    const double* col_ptr = matrix + static_cast<std::size_t>(column) * m;
    std::int8_t* x1_ptr   = X1 + static_cast<std::size_t>(column) * ldq;
    std::int8_t* x2_ptr   = X2 + static_cast<std::size_t>(column) * ldq;

    // Phase 1: Compute max |A_ij|
    double local_max1 = 0.0;
    for (int row = threadIdx.x; row < m; row += blockDim.x) {
        local_max1 = fmax(local_max1, fabs(col_ptr[row]));
    }
    double max1 = block_reduce_max(local_max1, smem);
    if (threadIdx.x == 0) smem[0] = max1;
    __syncthreads();
    max1 = smem[0];
    const double s1 = max1 > 0.0 ? max1 / 127.0 : 1.0;
    const double inv_s1 = max1 > 0.0 ? 127.0 / max1 : 0.0;
    if (threadIdx.x == 0) scale1[column] = s1;
    __syncthreads();

    // Phase 2: Quantize X1, compute residual locally in registers, and compute max |R_ij|
    double local_max2 = 0.0;
    for (int row = threadIdx.x; row < m; row += blockDim.x) {
        double val = col_ptr[row];
        double q1 = nearbyint(val * inv_s1);
        q1 = fmax(-127.0, fmin(127.0, q1));
        std::int8_t q1_i8 = static_cast<std::int8_t>(q1);
        x1_ptr[row] = q1_i8;

        // Residual in registers
        double res = val - static_cast<double>(q1_i8) * s1;
        local_max2 = fmax(local_max2, fabs(res));
    }
    double max2 = block_reduce_max(local_max2, smem);
    if (threadIdx.x == 0) smem[0] = max2;
    __syncthreads();
    max2 = smem[0];
    const double s2 = max2 > 0.0 ? max2 / 127.0 : 1.0;
    const double inv_s2 = max2 > 0.0 ? 127.0 / max2 : 0.0;
    if (threadIdx.x == 0) scale2[column] = s2;
    __syncthreads();

    // Phase 3: Quantize X2 using residual from X1 and A in registers
    for (int row = threadIdx.x; row < m; row += blockDim.x) {
        double val = col_ptr[row];
        std::int8_t q1_i8 = x1_ptr[row];
        double res = val - static_cast<double>(q1_i8) * s1;
        double q2 = nearbyint(res * inv_s2);
        q2 = fmax(-127.0, fmin(127.0, q2));
        x2_ptr[row] = static_cast<std::int8_t>(q2);
    }
}

__global__ void quantization_residual(const double* matrix, const std::int8_t* quantized,
                                      const double* scales, double* residual, int m, int n, int ldq) {
    const int column = blockIdx.x;
    if (column >= n) return;
    for (int row = threadIdx.x; row < m; row += blockDim.x) {
        const std::size_t index = static_cast<std::size_t>(column) * m + row;
        residual[index] = matrix[index] - static_cast<double>(quantized[static_cast<std::size_t>(column) * ldq + row]) * scales[column];
    }
}

__global__ void accumulate_integer_gram(const std::int32_t* integer_gram, double* gram,
                                        const double* row_scales, const double* column_scales, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && column < n)
        gram[static_cast<std::size_t>(column) * n + row] +=
            static_cast<double>(integer_gram[static_cast<std::size_t>(column) * n + row]) * row_scales[row] * column_scales[column];
}

__global__ void accumulate_integer_cross(const std::int32_t* cross, double* gram,
                                         const double* first_scales, const double* second_scales, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && column < n) {
        const double forward = static_cast<double>(cross[static_cast<std::size_t>(column) * n + row]);
        const double transpose = static_cast<double>(cross[static_cast<std::size_t>(row) * n + column]);
        gram[static_cast<std::size_t>(column) * n + row] +=
            first_scales[row] * second_scales[column] * forward +
            second_scales[row] * first_scales[column] * transpose;
    }
}

struct Workspace {
    Workspace(int rows, int columns, LibraryContext& context)
        : m(rows), n(columns), int8_leading_dimension((m + 15) & ~15),
          a(static_cast<std::size_t>(m) * n), q(a.size()), temporary_q(a.size()),
          residual(a.size()), x1(static_cast<std::size_t>(int8_leading_dimension) * n),
          x2(static_cast<std::size_t>(int8_leading_dimension) * n), gram(static_cast<std::size_t>(n) * n),
          gram_full(gram.size()), r(gram.size()), temporary_r(gram.size()), integer_gram(gram.size()),
          polar_h(gram.size()), polar_v(gram.size()), eig(n),
          norms(n), scale1(n), scale2(n), tau(n), info(1) {
        int potrf = 0, geqrf = 0, orgqr = 0, geqrf_nn = 0, orgqr_nn = 0, syevd = 0;
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(context.solver, CUBLAS_FILL_MODE_UPPER, n, gram.get(), n, &potrf));
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(context.solver, m, n, q.get(), m, &geqrf));
        CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(context.solver, m, n, n, q.get(), m, tau.get(), &orgqr));
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(context.solver, n, n, polar_h.get(), n, &geqrf_nn));
        CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(context.solver, n, n, n, polar_h.get(), n, tau.get(), &orgqr_nn));
        CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(context.solver, CUSOLVER_EIG_MODE_NOVECTOR,
                                                    CUBLAS_FILL_MODE_UPPER, n, gram.get(), n,
                                                    eig.get(), &syevd));
        work.allocate(std::max({potrf, geqrf, orgqr, geqrf_nn, orgqr_nn, syevd, 1}));
        // Active rows are overwritten by quantization; padded rows stay zero for
        // every trial and make the padded INT8 dot products mathematically neutral.
        CUDA_CHECK(cudaMemsetAsync(x1.get(), 0, x1.size() * sizeof(std::int8_t), context.stream));
        CUDA_CHECK(cudaMemsetAsync(x2.get(), 0, x2.size() * sizeof(std::int8_t), context.stream));
    }
    int m, n, int8_leading_dimension;
    DeviceBuffer<double> a, q, temporary_q, residual, gram, gram_full, r, temporary_r;
    DeviceBuffer<double> polar_h, polar_v, eig;
    DeviceBuffer<std::int8_t> x1, x2;
    DeviceBuffer<std::int32_t> integer_gram;
    DeviceBuffer<double> norms, scale1, scale2, tau, work;
    DeviceBuffer<int> info;
};

class EventTimer {
public:
    explicit EventTimer(cudaStream_t stream_, bool enabled_ = true) : stream(stream_), enabled(enabled_) {
        if (enabled) { CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); }
    }
    ~EventTimer() { if (start) cudaEventDestroy(start); if (stop) cudaEventDestroy(stop); }
    template <class Function> double measure(Function&& function) {
        if (!enabled) { function(); return 0.0; }
        CUDA_CHECK(cudaEventRecord(start, stream));
        function();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        return milliseconds;
    }
private:
    cudaStream_t stream;
    bool enabled;
    cudaEvent_t start = nullptr, stop = nullptr;
};

static int device_info(Workspace& workspace, LibraryContext& context) {
    int value = 0;
    CUDA_CHECK(cudaMemcpyAsync(&value, workspace.info.get(), sizeof(int), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));
    return value;
}

static void form_fp64_gram(Workspace& workspace, LibraryContext& context, const double* matrix) {
    const double one = 1.0, zero = 0.0;
    CUBLAS_CHECK(cublasDsyrk(context.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                             workspace.n, workspace.m, &one, matrix, workspace.m,
                             &zero, workspace.gram.get(), workspace.n));
}

static double orthogonality_from_gram(Workspace& workspace, LibraryContext& context) {
    const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
    form_gram_minus_identity<<<grid, block, 0, context.stream>>>(workspace.gram_full.get(), workspace.gram.get(), workspace.n);
    KERNEL_CHECK();
    double norm = 0.0;
    CUBLAS_CHECK(cublasDnrm2(context.blas, workspace.n * workspace.n, workspace.gram_full.get(), 1, &norm));
    return norm;
}

static double compute_orthogonality(Workspace& workspace, LibraryContext& context) {
    form_fp64_gram(workspace, context, workspace.q.get());
    return orthogonality_from_gram(workspace, context);
}

static double compute_residual(Workspace& workspace, LibraryContext& context, double input_norm) {
    const double one = 1.0, zero = 0.0, minus_one = -1.0;
    CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N, workspace.m, workspace.n, workspace.n,
                             &one, workspace.q.get(), workspace.m, workspace.r.get(), workspace.n,
                             &zero, workspace.temporary_q.get(), workspace.m));
    CUBLAS_CHECK(cublasDaxpy(context.blas, workspace.m * workspace.n, &minus_one,
                             workspace.a.get(), 1, workspace.temporary_q.get(), 1));
    double norm = 0.0;
    CUBLAS_CHECK(cublasDnrm2(context.blas, workspace.m * workspace.n, workspace.temporary_q.get(), 1, &norm));
    return norm / input_norm;
}

static void integer_gemm(Workspace& workspace, LibraryContext& context,
                         const std::int8_t* left, const std::int8_t* right) {
    const std::int32_t one = 1, zero = 0;
    CUBLAS_CHECK(cublasGemmEx(context.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                              workspace.n, workspace.n, workspace.int8_leading_dimension, &one,
                              left, CUDA_R_8I, workspace.int8_leading_dimension,
                              right, CUDA_R_8I, workspace.int8_leading_dimension,
                              &zero, workspace.integer_gram.get(), CUDA_R_32I, workspace.n,
                              CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
}

static void build_int8_gram(Workspace& workspace, LibraryContext& context, int levels, bool profile, PhaseTimes& times) {
    if (static_cast<std::int64_t>(workspace.int8_leading_dimension) * 127LL * 127LL > INT_MAX)
        throw std::runtime_error("INT32 dot-product overflow contract violated; split-K is required");
    EventTimer timer(context.stream, profile);
    const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);

    if (levels == 1) {
        times.quantization_ms += timer.measure([&] {
            quantize_columns<<<workspace.n, 256, 0, context.stream>>>(
                workspace.q.get(), workspace.x1.get(), workspace.scale1.get(), workspace.m, workspace.n,
                workspace.int8_leading_dimension);
            KERNEL_CHECK();
        });
        CUDA_CHECK(cudaMemsetAsync(workspace.gram.get(), 0, workspace.gram.size() * sizeof(double), context.stream));
        times.gram_ms += timer.measure([&] {
            integer_gemm(workspace, context, workspace.x1.get(), workspace.x1.get());
            accumulate_integer_gram<<<grid, block, 0, context.stream>>>(workspace.integer_gram.get(), workspace.gram.get(),
                                                                        workspace.scale1.get(), workspace.scale1.get(), workspace.n);
            KERNEL_CHECK();
        });
    } else {
        // Fused Two-Slice Quantization: single-pass in registers/shared memory without DRAM residual traffic
        times.quantization_ms += timer.measure([&] {
            fused_two_slice_quantize<<<workspace.n, 256, 0, context.stream>>>(
                workspace.q.get(), workspace.x1.get(), workspace.scale1.get(),
                workspace.x2.get(), workspace.scale2.get(),
                workspace.m, workspace.n, workspace.int8_leading_dimension);
            KERNEL_CHECK();
        });
        CUDA_CHECK(cudaMemsetAsync(workspace.gram.get(), 0, workspace.gram.size() * sizeof(double), context.stream));
        times.gram_ms += timer.measure([&] {
            integer_gemm(workspace, context, workspace.x1.get(), workspace.x1.get());
            accumulate_integer_gram<<<grid, block, 0, context.stream>>>(workspace.integer_gram.get(), workspace.gram.get(),
                                                                        workspace.scale1.get(), workspace.scale1.get(), workspace.n);
            integer_gemm(workspace, context, workspace.x1.get(), workspace.x2.get());
            accumulate_integer_cross<<<grid, block, 0, context.stream>>>(workspace.integer_gram.get(), workspace.gram.get(),
                                                                         workspace.scale1.get(), workspace.scale2.get(), workspace.n);
            integer_gemm(workspace, context, workspace.x2.get(), workspace.x2.get());
            accumulate_integer_gram<<<grid, block, 0, context.stream>>>(workspace.integer_gram.get(), workspace.gram.get(),
                                                                        workspace.scale2.get(), workspace.scale2.get(), workspace.n);
            KERNEL_CHECK();
        });
    }
}

static bool factor_and_apply_precomputed_gram(Workspace& workspace, LibraryContext& context,
                                               bool first_pass, bool need_r, double shift,
                                               bool profile, RunResult& result) {
    EventTimer timer(context.stream, profile);
    if (shift != 0.0)
        result.time.shift_ms += timer.measure([&] {
            add_diagonal<<<(workspace.n + 255) / 256, 256, 0, context.stream>>>(workspace.gram.get(), workspace.n, shift);
            KERNEL_CHECK();
        });
    result.time.cholesky_ms += timer.measure([&] {
        CUSOLVER_CHECK(cusolverDnDpotrf(context.solver, CUBLAS_FILL_MODE_UPPER, workspace.n,
                                        workspace.gram.get(), workspace.n, workspace.work.get(),
                                        static_cast<int>(workspace.work.size()), workspace.info.get()));
    });
    const int info = device_info(workspace, context);
    if (info != 0) {
        result.potrf_info = info;
        result.stop_reason = StopReason::PotrfFailure;
        return false;
    }
    const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
    if (need_r) {
        extract_upper<<<grid, block, 0, context.stream>>>(workspace.gram.get(), workspace.n, workspace.temporary_r.get(), workspace.n);
        KERNEL_CHECK();
    }
    const double one = 1.0;
    result.time.triangular_solve_ms += timer.measure([&] {
        CUBLAS_CHECK(cublasDtrsm(context.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                 CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, workspace.m, workspace.n,
                                 &one, workspace.gram.get(), workspace.n, workspace.q.get(), workspace.m));
    });
    if (need_r) {
        result.time.r_update_ms += timer.measure([&] {
            if (first_pass) {
                CUDA_CHECK(cudaMemcpyAsync(workspace.r.get(), workspace.temporary_r.get(),
                                           workspace.r.size() * sizeof(double), cudaMemcpyDeviceToDevice, context.stream));
            } else {
                CUBLAS_CHECK(cublasDtrmm(context.blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                         CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, workspace.n, workspace.n,
                                         &one, workspace.temporary_r.get(), workspace.n, workspace.r.get(), workspace.n,
                                         workspace.gram_full.get(), workspace.n));
                CUDA_CHECK(cudaMemcpyAsync(workspace.r.get(), workspace.gram_full.get(), workspace.r.size() * sizeof(double),
                                           cudaMemcpyDeviceToDevice, context.stream));
            }
        });
    }
    result.passes++;
    if (!first_pass) result.refinements++;
    return true;
}

static bool fp64_pass(Workspace& workspace, LibraryContext& context, bool first_pass,
                      bool need_r, double shift, bool profile, RunResult& result) {
    EventTimer timer(context.stream, profile);
    result.time.gram_ms += timer.measure([&] { form_fp64_gram(workspace, context, workspace.q.get()); });
    return factor_and_apply_precomputed_gram(workspace, context, first_pass, need_r, shift, profile, result);
}

static double published_shift_from_frobenius_norm(const Workspace& workspace, double norm) {
    const double unit_roundoff = 0.5 * std::numeric_limits<double>::epsilon();
    const double dimension_factor = static_cast<double>(workspace.m) * workspace.n +
                                    static_cast<double>(workspace.n) * (workspace.n + 1.0);
    return 11.0 * dimension_factor * unit_roundoff * norm * norm;
}

static double published_shift_frobenius(Workspace& workspace, LibraryContext& context) {
    double norm = 0.0;
    CUBLAS_CHECK(cublasDnrm2(context.blas, workspace.m * workspace.n, workspace.q.get(), 1, &norm));
    return published_shift_from_frobenius_norm(workspace, norm);
}

static bool prepare_input(Workspace& workspace, LibraryContext& context, const RunConfig& config,
                          RunResult& result) {
    EventTimer timer(context.stream, config.collect_phase_timings);
    result.time.preprocessing_ms += timer.measure([&] {
        CUDA_CHECK(cudaMemcpyAsync(workspace.q.get(), workspace.a.get(), workspace.a.size() * sizeof(double),
                                   cudaMemcpyDeviceToDevice, context.stream));
        if (config.equilibrate_columns) {
            column_norms<<<workspace.n, 256, 256 * sizeof(double), context.stream>>>(
                workspace.q.get(), workspace.norms.get(), workspace.m, workspace.n);
            divide_columns<<<workspace.n, 256, 0, context.stream>>>(workspace.q.get(), workspace.norms.get(), workspace.m, workspace.n);
            KERNEL_CHECK();
        }
    });
    return true;
}

// Fukaya et al. (2014), CholeskyQR2: exactly two unshifted CholeskyQR passes.
static RunResult run_choleskyqr2(Workspace& workspace, LibraryContext& context, const RunConfig& config) {
    RunResult result;
    EventTimer total(context.stream);
    result.time.total_ms = total.measure([&] {
        prepare_input(workspace, context, config, result);
        const bool need_r = config.output == OutputContract::QR;
        if (!fp64_pass(workspace, context, true, need_r, 0.0, config.collect_phase_timings, result)) return;
        if (!fp64_pass(workspace, context, false, need_r, 0.0, config.collect_phase_timings, result)) return;
        if (need_r && config.equilibrate_columns) {
            const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
            multiply_r_columns<<<grid, block, 0, context.stream>>>(workspace.r.get(), workspace.norms.get(), workspace.n);
            KERNEL_CHECK();
        }
        result.factorization_success = true;
        result.stop_reason = StopReason::FixedPasses;
    });
    return result;
}

// Fukaya et al. (2020), Algorithm 2.1: one shifted CholeskyQR pass.
static RunResult run_shifted_choleskyqr(Workspace& workspace, LibraryContext& context, const RunConfig& config) {
    RunResult result;
    EventTimer total(context.stream);
    result.time.total_ms = total.measure([&] {
        prepare_input(workspace, context, config, result);
        result.shift = published_shift_frobenius(workspace, context);
        const bool need_r = config.output == OutputContract::QR;
        if (!fp64_pass(workspace, context, true, need_r, result.shift, config.collect_phase_timings, result)) return;
        if (need_r && config.equilibrate_columns) {
            const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
            multiply_r_columns<<<grid, block, 0, context.stream>>>(workspace.r.get(), workspace.norms.get(), workspace.n);
            KERNEL_CHECK();
        }
        result.factorization_success = true;
        result.stop_reason = StopReason::FixedPasses;
    });
    return result;
}

// Fukaya et al. (2020), Algorithm 4.2: shifted pass followed by CholeskyQR2.
static RunResult run_shifted_choleskyqr3(Workspace& workspace, LibraryContext& context, const RunConfig& config) {
    RunResult result;
    EventTimer total(context.stream);
    result.time.total_ms = total.measure([&] {
        prepare_input(workspace, context, config, result);
        result.shift = published_shift_frobenius(workspace, context);
        const bool need_r = config.output == OutputContract::QR;
        if (!fp64_pass(workspace, context, true, need_r, result.shift, config.collect_phase_timings, result)) return;
        if (!fp64_pass(workspace, context, false, need_r, 0.0, config.collect_phase_timings, result)) return;
        if (!fp64_pass(workspace, context, false, need_r, 0.0, config.collect_phase_timings, result)) return;
        if (need_r && config.equilibrate_columns) {
            const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
            multiply_r_columns<<<grid, block, 0, context.stream>>>(workspace.r.get(), workspace.norms.get(), workspace.n);
            KERNEL_CHECK();
        }
        result.factorization_success = true;
        result.stop_reason = StopReason::FixedPasses;
    });
    return result;
}

// Reviewer control, not published CholeskyQR2: one pass plus tolerance-triggered refinements.
static RunResult run_adaptive(Workspace& workspace, LibraryContext& context, const RunConfig& config, int int8_levels) {
    RunResult result;
    EventTimer total(context.stream), timer(context.stream, config.collect_phase_timings);
    result.time.total_ms = total.measure([&] {
        prepare_input(workspace, context, config, result);
        const bool need_r = config.output == OutputContract::QR;
        if (int8_levels == 0) {
            if (!fp64_pass(workspace, context, true, need_r, 0.0, config.collect_phase_timings, result)) return;
        } else {
            build_int8_gram(workspace, context, int8_levels, config.collect_phase_timings, result.time);
            if (!factor_and_apply_precomputed_gram(workspace, context, true, need_r, 0.0,
                                                   config.collect_phase_timings, result)) return;
        }
        result.time.stopping_check_ms += timer.measure([&] {
            form_fp64_gram(workspace, context, workspace.q.get());
            result.initial_orthogonality_fro = orthogonality_from_gram(workspace, context);
        });
        double current = result.initial_orthogonality_fro;
        while (current > config.orthogonality_tolerance && result.refinements < config.maximum_refinements) {
            if (!factor_and_apply_precomputed_gram(workspace, context, false, need_r, 0.0,
                                                   config.collect_phase_timings, result)) return;
            result.time.stopping_check_ms += timer.measure([&] {
                form_fp64_gram(workspace, context, workspace.q.get());
                current = orthogonality_from_gram(workspace, context);
            });
        }
        result.accuracy_target_met = std::isfinite(current) && current <= config.orthogonality_tolerance;
        result.stop_reason = result.accuracy_target_met ? StopReason::ToleranceMet : StopReason::PassLimit;
        if (need_r && config.equilibrate_columns) {
            const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
            multiply_r_columns<<<grid, block, 0, context.stream>>>(workspace.r.get(), workspace.norms.get(), workspace.n);
            KERNEL_CHECK();
        }
        result.factorization_success = true;
    });
    return result;
}

// Fixed classical CholeskyQR baseline with INT8 Gram initialization. Unlike the
// adaptive methods, this performs no orthogonality-driven second pass.
static RunResult run_int8_choleskyqr(Workspace& workspace, LibraryContext& context,
                                     const RunConfig& config, int int8_levels) {
    RunResult result;
    EventTimer total(context.stream);
    result.time.total_ms = total.measure([&] {
        prepare_input(workspace, context, config, result);
        build_int8_gram(workspace, context, int8_levels, config.collect_phase_timings, result.time);
        const bool need_r = config.output == OutputContract::QR;
        if (!factor_and_apply_precomputed_gram(workspace, context, true, need_r, 0.0,
                                               config.collect_phase_timings, result)) return;
        if (need_r && config.equilibrate_columns) {
            const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
            multiply_r_columns<<<grid, block, 0, context.stream>>>(workspace.r.get(), workspace.norms.get(), workspace.n);
            KERNEL_CHECK();
        }
        result.factorization_success = true;
        result.stop_reason = StopReason::FixedPasses;
    });
    return result;
}

static bool gram_eigen_extremes(Workspace& workspace, LibraryContext& context,
                                double& lambda_min, double& lambda_max, RunResult& result) {
    CUDA_CHECK(cudaMemcpyAsync(workspace.gram_full.get(), workspace.gram.get(),
                               workspace.gram.size() * sizeof(double), cudaMemcpyDeviceToDevice,
                               context.stream));
    CUSOLVER_CHECK(cusolverDnDsyevd(context.solver, CUSOLVER_EIG_MODE_NOVECTOR,
                                    CUBLAS_FILL_MODE_UPPER, workspace.n,
                                    workspace.gram_full.get(), workspace.n,
                                    workspace.eig.get(), workspace.work.get(),
                                    static_cast<int>(workspace.work.size()), workspace.info.get()));
    const int info = device_info(workspace, context);
    if (info != 0) {
        result.potrf_info = info;
        result.stop_reason = StopReason::PotrfFailure;
        return false;
    }
    double values[2] = {0.0, 0.0};
    CUDA_CHECK(cudaMemcpyAsync(&values[0], workspace.eig.get(), sizeof(double), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaMemcpyAsync(&values[1], workspace.eig.get() + workspace.n - 1, sizeof(double), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));
    lambda_min = values[0];
    lambda_max = values[1];
    return true;
}

static std::vector<double> chebyshev_invsqrt_coefficients(double epsilon, int terms) {
    const double pi = std::acos(-1.0);
    std::vector<double> coefficients(terms, 0.0);
    for (int k = 0; k < terms; ++k) {
        double sum = 0.0;
        for (int j = 0; j < terms; ++j) {
            const double theta = pi * (static_cast<double>(j) + 0.5) / terms;
            const double x = 1.0 + epsilon * std::cos(theta);
            sum += (1.0 / std::sqrt(x)) * std::cos(k * theta);
        }
        coefficients[k] = 2.0 * sum / terms;
    }
    coefficients[0] *= 0.5;
    return coefficients;
}

static bool evaluate_chebyshev_invsqrt(Workspace& workspace, LibraryContext& context,
                                       const RunConfig& config, double lambda_min, double lambda_max,
                                       double& mu, double*& w_inverse_sqrt, RunResult& result) {
    if (!(lambda_min > 0.0) || !(lambda_max >= lambda_min) ||
        !std::isfinite(lambda_min) || !std::isfinite(lambda_max)) {
        result.stop_reason = StopReason::InvalidInput;
        return false;
    }
    mu = 0.5 * (lambda_max + lambda_min);
    const double epsilon = (lambda_max - lambda_min) / (lambda_max + lambda_min);
    if (epsilon < 1.0e-14) {
        result.chebyshev_degree = 0;
        const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
        set_identity<<<grid, block, 0, context.stream>>>(workspace.polar_h.get(), workspace.n);
        KERNEL_CHECK();
        w_inverse_sqrt = workspace.polar_h.get();
        return true;
    }
    const double kappa = std::sqrt(lambda_max / lambda_min);
    const double rate = (kappa + 1.0) / (kappa - 1.0);
    const double degree_real = std::log(2.0 * std::sqrt(lambda_max) / config.orthogonality_tolerance) /
                               std::log(rate);
    const int degree = std::max(1, static_cast<int>(std::ceil(degree_real)));
    result.chebyshev_degree = degree;
    if (degree > config.maximum_chebyshev_degree) {
        result.stop_reason = StopReason::ChebyshevDegreeLimit;
        return false;
    }

    const std::vector<double> coefficients = chebyshev_invsqrt_coefficients(epsilon, degree + 1);
    const double inverse_mu = 1.0 / mu;
    const double inverse_epsilon = 1.0 / epsilon;
    // DSYRK leaves only the upper triangle authoritative. Clenshaw uses full
    // GEMMs, so complete the symmetric matrix once before the polynomial work.
    {
        const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
        symmetrize_upper_to_full<<<grid, block, 0, context.stream>>>(workspace.gram.get(), workspace.n);
        KERNEL_CHECK();
    }
    CUDA_CHECK(cudaMemcpyAsync(workspace.temporary_r.get(), workspace.gram.get(),
                               workspace.gram.size() * sizeof(double), cudaMemcpyDeviceToDevice,
                               context.stream));
    CUBLAS_CHECK(cublasDscal(context.blas, workspace.n * workspace.n, &inverse_mu,
                             workspace.temporary_r.get(), 1));
    subtract_identity_full<<<(workspace.n + 255) / 256, 256, 0, context.stream>>>(
        workspace.temporary_r.get(), workspace.n);
    KERNEL_CHECK();
    CUBLAS_CHECK(cublasDscal(context.blas, workspace.n * workspace.n, &inverse_epsilon,
                             workspace.temporary_r.get(), 1));

    CUDA_CHECK(cudaMemsetAsync(workspace.gram_full.get(), 0,
                               workspace.gram_full.size() * sizeof(double), context.stream));
    CUDA_CHECK(cudaMemsetAsync(workspace.polar_v.get(), 0,
                               workspace.polar_v.size() * sizeof(double), context.stream));
    double* next = workspace.gram_full.get();
    double* next2 = workspace.polar_v.get();
    double* out = workspace.polar_h.get();
    const double one = 1.0, zero = 0.0, minus_one = -1.0, two = 2.0;

    for (int k = degree; k >= 0; --k) {
        CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 workspace.n, workspace.n, workspace.n,
                                 &one, workspace.temporary_r.get(), workspace.n,
                                 next, workspace.n, &zero, out, workspace.n));
        CUBLAS_CHECK(cublasDscal(context.blas, workspace.n * workspace.n, &two, out, 1));
        CUBLAS_CHECK(cublasDaxpy(context.blas, workspace.n * workspace.n, &minus_one, next2, 1, out, 1));
        add_diagonal<<<(workspace.n + 255) / 256, 256, 0, context.stream>>>(
            out, workspace.n, coefficients[k]);
        KERNEL_CHECK();
        double* old_next2 = next2;
        next2 = next;
        next = out;
        out = old_next2;
    }

    // W = B_0 - Bhat B_1. Use temporary_r as the non-aliased destination:
    // copying B0 directly into polar_h can overwrite the just-computed Bhat*B1
    // for some degree parities.
    CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                             workspace.n, workspace.n, workspace.n,
                             &one, workspace.temporary_r.get(), workspace.n,
                             next2, workspace.n, &zero, out, workspace.n));
    CUDA_CHECK(cudaMemcpyAsync(workspace.temporary_r.get(), next,
                               workspace.temporary_r.size() * sizeof(double),
                               cudaMemcpyDeviceToDevice, context.stream));
    CUBLAS_CHECK(cublasDaxpy(context.blas, workspace.n * workspace.n, &minus_one,
                             out, 1, workspace.temporary_r.get(), 1));
    CUDA_CHECK(cudaMemcpyAsync(workspace.polar_h.get(), workspace.temporary_r.get(),
                               workspace.polar_h.size() * sizeof(double),
                               cudaMemcpyDeviceToDevice, context.stream));
    w_inverse_sqrt = workspace.polar_h.get();

    // Certificate: ||W G W - mu I||_F / mu, evaluated only on the
    // small Gram matrix. This records the actual inverse-square-root quality.
    const double minus_mu = -mu;
    CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                             workspace.n, workspace.n, workspace.n,
                             &one, workspace.gram.get(), workspace.n,
                             w_inverse_sqrt, workspace.n,
                             &zero, workspace.gram_full.get(), workspace.n));
    CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                             workspace.n, workspace.n, workspace.n,
                             &one, w_inverse_sqrt, workspace.n,
                             workspace.gram_full.get(), workspace.n,
                             &zero, workspace.temporary_r.get(), workspace.n));
    add_diagonal<<<(workspace.n + 255) / 256, 256, 0, context.stream>>>(
        workspace.temporary_r.get(), workspace.n, minus_mu);
    KERNEL_CHECK();
    double certificate = 0.0;
    CUBLAS_CHECK(cublasDnrm2(context.blas, workspace.n * workspace.n,
                             workspace.temporary_r.get(), 1, &certificate));
    result.inverse_sqrt_residual_fro = certificate / mu;
    return true;
}

static bool small_qr_apply(Workspace& workspace, LibraryContext& context,
                           double* small_matrix, const RunConfig& config, RunResult& result) {
    EventTimer timer(context.stream, config.collect_phase_timings);
    result.time.small_qr_ms += timer.measure([&] {
        CUSOLVER_CHECK(cusolverDnDgeqrf(context.solver, workspace.n, workspace.n,
                                        small_matrix, workspace.n,
                                        workspace.tau.get(), workspace.work.get(),
                                        static_cast<int>(workspace.work.size()), workspace.info.get()));
    });
    int info = device_info(workspace, context);
    if (info != 0) {
        result.potrf_info = info;
        result.stop_reason = StopReason::PotrfFailure;
        return false;
    }

    result.time.small_qr_ms += timer.measure([&] {
        const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
        extract_upper<<<grid, block, 0, context.stream>>>(small_matrix, workspace.n,
                                                          workspace.r.get(), workspace.n);
        KERNEL_CHECK();
        CUSOLVER_CHECK(cusolverDnDorgqr(context.solver, workspace.n, workspace.n, workspace.n,
                                        small_matrix, workspace.n, workspace.tau.get(),
                                        workspace.work.get(), static_cast<int>(workspace.work.size()),
                                        workspace.info.get()));
    });
    info = device_info(workspace, context);
    if (info != 0) {
        result.potrf_info = info;
        result.stop_reason = StopReason::PotrfFailure;
        return false;
    }

    const double one = 1.0, zero = 0.0;
    result.time.ns_tall_gemm_ms += timer.measure([&] {
        CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 workspace.m, workspace.n, workspace.n,
                                 &one, workspace.q.get(), workspace.m,
                                 small_matrix, workspace.n,
                                 &zero, workspace.temporary_q.get(), workspace.m));
    });
    std::swap(workspace.q, workspace.temporary_q);

    if (config.equilibrate_columns) {
        const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
        multiply_r_columns<<<grid, block, 0, context.stream>>>(workspace.r.get(), workspace.norms.get(), workspace.n);
        KERNEL_CHECK();
    }
    return true;
}

// Cholesky-free inverse-square-root method. L2 builds M ~= A^T A. Coupled
// Newton-Schulz computes M^{-1/2}; Q = A M^{-1/2} is optionally polished by
// guarded polar Newton-Schulz. The Q-only variant stops there. The QR variant
// uses the coupled iteration's small V ~= M^{1/2} and factors only that n-by-n
// matrix to recover the triangular factor.
static RunResult run_int8_l2_polar_ns(Workspace& workspace, LibraryContext& context,
                                      const RunConfig& config) {
    RunResult result;
    const bool complete_qr = (config.algorithm == Algorithm::Int8L2PolarNSQR);
    if (complete_qr != (config.output == OutputContract::QR)) {
        result.stop_reason = StopReason::InvalidInput;
        return result;
    }

    EventTimer total(context.stream), timer(context.stream, config.collect_phase_timings);
    result.time.total_ms = total.measure([&] {
        prepare_input(workspace, context, config, result);
        build_int8_gram(workspace, context, 2, config.collect_phase_timings, result.time);

        // Form F2 = F1-X2*S2 in place and report its relative Frobenius norm.
        // It is a diagnostic certificate, not an added regularization: using the
        // conservative Gram-error bound as a shift over-regularizes Q and makes
        // the subsequent tall polar iteration unnecessarily expensive.
        quantization_residual<<<workspace.n, 256, 0, context.stream>>>(
            workspace.residual.get(), workspace.x2.get(), workspace.scale2.get(),
            workspace.residual.get(), workspace.m, workspace.n,
            workspace.int8_leading_dimension);
        KERNEL_CHECK();
        double input_norm = 0.0, residual_norm = 0.0;
        CUBLAS_CHECK(cublasDnrm2(context.blas, workspace.m * workspace.n,
                                 workspace.q.get(), 1, &input_norm));
        CUBLAS_CHECK(cublasDnrm2(context.blas, workspace.m * workspace.n,
                                 workspace.residual.get(), 1, &residual_norm));
        result.quantization_residual_fro = residual_norm / input_norm;
        result.shift = published_shift_from_frobenius_norm(workspace, input_norm);
        result.time.shift_ms += timer.measure([&] {
            add_diagonal<<<(workspace.n + 255) / 256, 256, 0, context.stream>>>(
                workspace.gram.get(), workspace.n, result.shift);
            KERNEL_CHECK();
        });

        double normalization = 0.0;
        CUBLAS_CHECK(cublasDnrm2(context.blas, workspace.n * workspace.n,
                                 workspace.gram.get(), 1, &normalization));
        if (!(normalization > 0.0) || !std::isfinite(normalization)) {
            result.stop_reason = StopReason::InvalidInput;
            return;
        }

        const double inverse_normalization = 1.0 / normalization;
        CUBLAS_CHECK(cublasDscal(context.blas, workspace.n * workspace.n,
                                 &inverse_normalization, workspace.gram.get(), 1));
        const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
        set_identity<<<grid, block, 0, context.stream>>>(workspace.r.get(), workspace.n);
        KERNEL_CHECK();

        double* y = workspace.gram.get();
        double* z = workspace.r.get();
        double* transform = workspace.gram_full.get();
        double* spare_y = workspace.temporary_r.get();
        double* spare_z = workspace.temporary_q.get();
        const double one = 1.0, zero = 0.0;
        bool inverse_converged = false;

        for (int iteration = 0; iteration <= config.maximum_inverse_sqrt_iterations; ++iteration) {
            result.time.ns_small_gemm_ms += timer.measure([&] {
                CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                         workspace.n, workspace.n, workspace.n,
                                         &one, z, workspace.n, y, workspace.n,
                                         &zero, transform, workspace.n));
            });
            subtract_identity_full<<<(workspace.n + 255) / 256, 256, 0, context.stream>>>(
                transform, workspace.n);
            KERNEL_CHECK();
            CUBLAS_CHECK(cublasDnrm2(context.blas, workspace.n * workspace.n,
                                     transform, 1, &result.inverse_sqrt_residual_fro));
            if (std::isfinite(result.inverse_sqrt_residual_fro) &&
                result.inverse_sqrt_residual_fro <= config.inverse_sqrt_tolerance) {
                inverse_converged = true;
                break;
            }
            if (iteration == config.maximum_inverse_sqrt_iterations ||
                !std::isfinite(result.inverse_sqrt_residual_fro)) break;

            inverse_sqrt_transform_from_residual<<<grid, block, 0, context.stream>>>(
                transform, workspace.n);
            KERNEL_CHECK();
            double* next_y = spare_y;
            double* next_z = spare_z;
            result.time.ns_small_gemm_ms += timer.measure([&] {
                CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                         workspace.n, workspace.n, workspace.n,
                                         &one, y, workspace.n, transform, workspace.n,
                                         &zero, next_y, workspace.n));
                CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                         workspace.n, workspace.n, workspace.n,
                                         &one, transform, workspace.n, z, workspace.n,
                                         &zero, next_z, workspace.n));
            });
            spare_y = y;
            spare_z = z;
            y = next_y;
            z = next_z;
            result.inverse_sqrt_iterations++;
        }

        if (!inverse_converged) {
            result.stop_reason = StopReason::InverseSqrtLimit;
            return;
        }

        // temporary_q is the only m-by-n output buffer. If it currently holds Z,
        // preserve Z in an available n-by-n buffer before the tall GEMM.
        if (z == workspace.temporary_q.get()) {
            double* preserved_z = workspace.temporary_r.get();
            if (preserved_z == y) preserved_z = workspace.r.get();
            CUDA_CHECK(cudaMemcpyAsync(preserved_z, z, workspace.gram.size() * sizeof(double),
                                       cudaMemcpyDeviceToDevice, context.stream));
            z = preserved_z;
        }
        if (complete_qr) {
            // Y is the small-matrix square root V ~= (G/normalization)^{1/2}.
            // Preserve it before tall operations overwrite the small work buffers.
            CUDA_CHECK(cudaMemcpyAsync(workspace.polar_h.get(), y,
                                       workspace.polar_h.size() * sizeof(double),
                                       cudaMemcpyDeviceToDevice, context.stream));
        }
        const double inverse_root_normalization = 1.0 / std::sqrt(normalization);
        result.time.ns_tall_gemm_ms += timer.measure([&] {
            CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                     workspace.m, workspace.n, workspace.n,
                                     &inverse_root_normalization, workspace.q.get(), workspace.m,
                                     z, workspace.n, &zero, workspace.temporary_q.get(), workspace.m));
        });
        std::swap(workspace.q, workspace.temporary_q);
        if (complete_qr) {
            const double root_normalization = std::sqrt(normalization);
            CUBLAS_CHECK(cublasDscal(context.blas, workspace.n * workspace.n,
                                     &root_normalization, workspace.polar_h.get(), 1));
            if (!small_qr_apply(workspace, context, workspace.polar_h.get(), config, result)) return;
        }
        result.factorization_success = true;
        result.stop_reason = StopReason::FixedPasses;
    });
    return result;
}

// GCOF direct Gram-Chebyshev method. The small Gram matrix is the only object
// used to approximate G^{-1/2}; there is no Newton iteration and no polar polish.
static RunResult run_int8_l2_polar_chebyshev(Workspace& workspace, LibraryContext& context,
                                             const RunConfig& config) {
    RunResult result;
    const bool complete_qr = (config.algorithm == Algorithm::Int8L2PolarChebyshevQR);
    if (complete_qr != (config.output == OutputContract::QR)) {
        result.stop_reason = StopReason::InvalidInput;
        return result;
    }
    EventTimer total(context.stream), timer(context.stream, config.collect_phase_timings);
    result.time.total_ms = total.measure([&] {
        prepare_input(workspace, context, config, result);
        build_int8_gram(workspace, context, 2, config.collect_phase_timings, result.time);
        double lambda_min = 0.0, lambda_max = 0.0, mu = 0.0;
        if (!gram_eigen_extremes(workspace, context, lambda_min, lambda_max, result)) return;
        double* w_inverse_sqrt = nullptr;
        if (!evaluate_chebyshev_invsqrt(workspace, context, config, lambda_min, lambda_max,
                                        mu, w_inverse_sqrt, result)) return;

        if (complete_qr) {
            // V = G W ~= G^{1/2}; it is the only small matrix used for R recovery.
            const double one = 1.0, zero = 0.0, inverse_root_mu_small = 1.0 / std::sqrt(mu);
            result.time.ns_small_gemm_ms += timer.measure([&] {
                CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                         workspace.n, workspace.n, workspace.n,
                                         &one, workspace.gram.get(), workspace.n,
                                         w_inverse_sqrt, workspace.n,
                                         &zero, workspace.temporary_r.get(), workspace.n));
                CUBLAS_CHECK(cublasDscal(context.blas, workspace.n * workspace.n,
                                         &inverse_root_mu_small, workspace.temporary_r.get(), 1));
            });
        }

        const double zero = 0.0, inverse_root_mu = 1.0 / std::sqrt(mu);
        result.time.ns_tall_gemm_ms += timer.measure([&] {
            CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                     workspace.m, workspace.n, workspace.n,
                                     &inverse_root_mu, workspace.q.get(), workspace.m,
                                     w_inverse_sqrt, workspace.n,
                                     &zero, workspace.temporary_q.get(), workspace.m));
        });
        std::swap(workspace.q, workspace.temporary_q);

        if (complete_qr &&
            !small_qr_apply(workspace, context, workspace.temporary_r.get(), config, result)) return;
        result.factorization_success = true;
        result.stop_reason = StopReason::FixedPasses;
    });
    return result;
}

// NIMA / GCOF refinement: start from an INT8 CholeskyQR basis Q0,R0, then
// apply one a-priori Chebyshev inverse-square-root correction to H=Q0^T Q0.
static RunResult run_int8_l2_nima_refinement(Workspace& workspace, LibraryContext& context,
                                             const RunConfig& config) {
    RunResult result;
    const bool complete_qr = (config.algorithm == Algorithm::Int8L2NIMARefinementQR);
    if (complete_qr != (config.output == OutputContract::QR)) {
        result.stop_reason = StopReason::InvalidInput;
        return result;
    }
    EventTimer total(context.stream), timer(context.stream, config.collect_phase_timings);
    result.time.total_ms = total.measure([&] {
        prepare_input(workspace, context, config, result);
        build_int8_gram(workspace, context, 2, config.collect_phase_timings, result.time);
        if (!factor_and_apply_precomputed_gram(workspace, context, true, complete_qr, 0.0,
                                               config.collect_phase_timings, result)) return;
        result.passes++;

        result.time.stopping_check_ms += timer.measure([&] {
            form_fp64_gram(workspace, context, workspace.q.get());
            result.initial_orthogonality_fro = orthogonality_from_gram(workspace, context);
        });
        if (std::isfinite(result.initial_orthogonality_fro) &&
            result.initial_orthogonality_fro <= config.orthogonality_tolerance) {
            result.accuracy_target_met = true;
            result.stop_reason = StopReason::ToleranceMet;
            if (complete_qr && config.equilibrate_columns) {
                const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
                multiply_r_columns<<<grid, block, 0, context.stream>>>(workspace.r.get(), workspace.norms.get(), workspace.n);
                KERNEL_CHECK();
            }
            result.factorization_success = true;
            return;
        }

        double lambda_min = 0.0, lambda_max = 0.0, mu = 0.0;
        if (!gram_eigen_extremes(workspace, context, lambda_min, lambda_max, result)) return;
        double* w_inverse_sqrt = nullptr;
        if (!evaluate_chebyshev_invsqrt(workspace, context, config, lambda_min, lambda_max,
                                        mu, w_inverse_sqrt, result)) return;

        const double one = 1.0, zero = 0.0, inverse_root_mu = 1.0 / std::sqrt(mu);
        result.time.ns_tall_gemm_ms += timer.measure([&] {
            CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                     workspace.m, workspace.n, workspace.n,
                                     &inverse_root_mu, workspace.q.get(), workspace.m,
                                     w_inverse_sqrt, workspace.n,
                                     &zero, workspace.temporary_q.get(), workspace.m));
        });
        std::swap(workspace.q, workspace.temporary_q);

        if (complete_qr) {
            // V = H W ~= H^{1/2}; R_pre = V R0. This overwrites polar_h only
            // after the tall Q update above has consumed W.
            result.time.ns_small_gemm_ms += timer.measure([&] {
                CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                         workspace.n, workspace.n, workspace.n,
                                         &one, workspace.gram.get(), workspace.n,
                                         w_inverse_sqrt, workspace.n,
                                         &zero, workspace.temporary_r.get(), workspace.n));
                CUBLAS_CHECK(cublasDscal(context.blas, workspace.n * workspace.n,
                                         &inverse_root_mu, workspace.temporary_r.get(), 1));
                CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                         workspace.n, workspace.n, workspace.n,
                                         &one, workspace.temporary_r.get(), workspace.n,
                                         workspace.r.get(), workspace.n,
                                         &zero, workspace.polar_h.get(), workspace.n));
            });
        }

        result.passes++;
        result.refinements++;

        if (complete_qr &&
            !small_qr_apply(workspace, context, workspace.polar_h.get(), config, result)) return;
        result.factorization_success = true;
        result.stop_reason = StopReason::FixedPasses;
    });
    return result;
}

static RunResult run_householder(Workspace& workspace, LibraryContext& context, const RunConfig& config) {
    RunResult result;
    EventTimer total(context.stream), timer(context.stream, config.collect_phase_timings);
    result.time.total_ms = total.measure([&] {
        prepare_input(workspace, context, config, result);
        result.time.householder_factorization_ms += timer.measure([&] {
            CUSOLVER_CHECK(cusolverDnDgeqrf(context.solver, workspace.m, workspace.n, workspace.q.get(), workspace.m,
                                            workspace.tau.get(), workspace.work.get(), static_cast<int>(workspace.work.size()),
                                            workspace.info.get()));
        });
        if (device_info(workspace, context) != 0) return;
        if (config.output == OutputContract::QR) {
            const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
            extract_upper<<<grid, block, 0, context.stream>>>(workspace.q.get(), workspace.m, workspace.r.get(), workspace.n);
            KERNEL_CHECK();
        }
        result.time.explicit_q_ms += timer.measure([&] {
            CUSOLVER_CHECK(cusolverDnDorgqr(context.solver, workspace.m, workspace.n, workspace.n,
                                            workspace.q.get(), workspace.m, workspace.tau.get(), workspace.work.get(),
                                            static_cast<int>(workspace.work.size()), workspace.info.get()));
        });
        if (device_info(workspace, context) != 0) return;
        if (config.output == OutputContract::QR && config.equilibrate_columns) {
            const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
            multiply_r_columns<<<grid, block, 0, context.stream>>>(workspace.r.get(), workspace.norms.get(), workspace.n);
            KERNEL_CHECK();
        }
        result.factorization_success = true;
        result.passes = 1;
        result.stop_reason = StopReason::FixedPasses;
    });
    return result;
}

// INT8 L2 adaptive with shift/retry on Cholesky failure.
static RunResult run_int8_l2_shift_retry(Workspace& workspace, LibraryContext& context, const RunConfig& config) {
    RunResult result;
    EventTimer total(context.stream), timer(context.stream, config.collect_phase_timings);
    result.time.total_ms = total.measure([&] {
        prepare_input(workspace, context, config, result);
        const bool need_r = config.output == OutputContract::QR;
        build_int8_gram(workspace, context, 2, config.collect_phase_timings, result.time);
        CUDA_CHECK(cudaMemcpyAsync(workspace.gram_full.get(), workspace.gram.get(),
                                   workspace.gram.size() * sizeof(double), cudaMemcpyDeviceToDevice,
                                   context.stream));
        double current_shift = 0.0;
        for (int attempt = 0; attempt <= config.l2_shift_retries; ++attempt) {
            if (attempt > 0) {
                if (current_shift == 0.0) current_shift = config.l2_shift;
                else current_shift *= 2.0;
                result.time.shift_ms += timer.measure([&] {
                    CUDA_CHECK(cudaMemcpyAsync(workspace.gram.get(), workspace.gram_full.get(),
                                               workspace.gram.size() * sizeof(double), cudaMemcpyDeviceToDevice,
                                               context.stream));
                    add_diagonal<<<(workspace.n + 255) / 256, 256, 0, context.stream>>>(workspace.gram.get(), workspace.n, current_shift);
                    KERNEL_CHECK();
                });
                result.shift = current_shift;
            }
            if (factor_and_apply_precomputed_gram(workspace, context, true, need_r, 0.0,
                                                   config.collect_phase_timings, result)) {
                result.passes = 1;
                break;
            }
            if (attempt == config.l2_shift_retries) return;
        }
        result.time.stopping_check_ms += timer.measure([&] {
            form_fp64_gram(workspace, context, workspace.q.get());
            result.initial_orthogonality_fro = orthogonality_from_gram(workspace, context);
        });
        double current = result.initial_orthogonality_fro;
        while (current > config.orthogonality_tolerance && result.refinements < config.maximum_refinements) {
            if (!factor_and_apply_precomputed_gram(workspace, context, false, need_r, 0.0,
                                                   config.collect_phase_timings, result)) return;
            result.time.stopping_check_ms += timer.measure([&] {
                form_fp64_gram(workspace, context, workspace.q.get());
                current = orthogonality_from_gram(workspace, context);
            });
        }
        result.accuracy_target_met = std::isfinite(current) && current <= config.orthogonality_tolerance;
        result.stop_reason = result.accuracy_target_met ? StopReason::ToleranceMet : StopReason::PassLimit;
        if (need_r && config.equilibrate_columns) {
            const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
            multiply_r_columns<<<grid, block, 0, context.stream>>>(workspace.r.get(), workspace.norms.get(), workspace.n);
            KERNEL_CHECK();
        }
        result.factorization_success = true;
    });
    return result;
}

// Residual-energy ridge (historical artifact name: CertifiedRidge).
// The shift s = (m / 12) * ||scale2||_2^2 is the total residual energy
// predicted by an independent uniform-error model. It is a heuristic,
// not the deterministic residual bound (m / 4) * ||scale2||_2^2 and not
// a finite-precision Cholesky certificate.
static RunResult run_int8_l2_certified_ridge(Workspace& workspace, LibraryContext& context, const RunConfig& config) {
    RunResult result;
    const bool complete_qr = (config.algorithm == Algorithm::Int8L2CertifiedRidgeQR);
    if (complete_qr != (config.output == OutputContract::QR)) {
        result.stop_reason = StopReason::InvalidInput;
        return result;
    }
    EventTimer total(context.stream), timer(context.stream, config.collect_phase_timings);
    result.time.total_ms = total.measure([&] {
        prepare_input(workspace, context, config, result);
        const bool need_r = config.output == OutputContract::QR;
        build_int8_gram(workspace, context, 2, config.collect_phase_timings, result.time);

        // Compute the residual-energy shift s = (m / 12) * ||scale2||_2^2.
        double scale2_dot = 0.0;
        CUBLAS_CHECK(cublasDdot(context.blas, workspace.n, workspace.scale2.get(), 1, workspace.scale2.get(), 1, &scale2_dot));
        double f_bar_sq = (static_cast<double>(workspace.m) / 12.0) * scale2_dot;
        result.shift = f_bar_sq;

        if (!factor_and_apply_precomputed_gram(workspace, context, true, need_r, f_bar_sq,
                                               config.collect_phase_timings, result)) return;
        result.passes = 1;

        result.time.stopping_check_ms += timer.measure([&] {
            form_fp64_gram(workspace, context, workspace.q.get());
            result.initial_orthogonality_fro = orthogonality_from_gram(workspace, context);
        });

        double current = result.initial_orthogonality_fro;
        while (current > config.orthogonality_tolerance && result.refinements < config.maximum_refinements) {
            if (!factor_and_apply_precomputed_gram(workspace, context, false, need_r, 0.0,
                                                   config.collect_phase_timings, result)) return;
            result.time.stopping_check_ms += timer.measure([&] {
                form_fp64_gram(workspace, context, workspace.q.get());
                current = orthogonality_from_gram(workspace, context);
            });
        }

        result.accuracy_target_met = std::isfinite(current) && current <= config.orthogonality_tolerance;
        result.stop_reason = result.accuracy_target_met ? StopReason::ToleranceMet : StopReason::PassLimit;
        if (need_r && config.equilibrate_columns) {
            const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
            multiply_r_columns<<<grid, block, 0, context.stream>>>(workspace.r.get(), workspace.norms.get(), workspace.n);
            KERNEL_CHECK();
        }
        result.factorization_success = true;
    });
    return result;
}

// Theorem 4.7: Triangular Defect-Correction Polish
// Computes initial factors Q0, R0 via 1-pass INT8-L2 CholeskyQR.
// Computes E = Q0^T Q0 - I, decomposes into T + T^T = E, sets P = I - T.
// Updates Q+ = Q * P and R+ = P^{-1} R.
// Completely bypasses potrf in the polish pass and strictly preserves upper-triangularity of R.
static RunResult run_int8_l2_triangular_polish(Workspace& workspace, LibraryContext& context,
                                               const RunConfig& config) {
    RunResult result;
    const bool complete_qr = (config.algorithm == Algorithm::Int8L2TriangularPolishQR);
    if (complete_qr != (config.output == OutputContract::QR)) {
        result.stop_reason = StopReason::InvalidInput;
        return result;
    }
    EventTimer total(context.stream), timer(context.stream, config.collect_phase_timings);
    result.time.total_ms = total.measure([&] {
        prepare_input(workspace, context, config, result);
        const bool need_r = config.output == OutputContract::QR;

        // Pass 1: INT8-L2 CholeskyQR
        build_int8_gram(workspace, context, 2, config.collect_phase_timings, result.time);
        if (!factor_and_apply_precomputed_gram(workspace, context, true, need_r, 0.0,
                                               config.collect_phase_timings, result)) return;
        result.passes = 1;

        // Form Q0^T Q0 and measure initial defect E = Q0^T Q0 - I
        result.time.stopping_check_ms += timer.measure([&] {
            form_fp64_gram(workspace, context, workspace.q.get());
            result.initial_orthogonality_fro = orthogonality_from_gram(workspace, context);
        });

        if (std::isfinite(result.initial_orthogonality_fro) &&
            result.initial_orthogonality_fro <= config.orthogonality_tolerance) {
            result.accuracy_target_met = true;
            result.stop_reason = StopReason::ToleranceMet;
            if (need_r && config.equilibrate_columns) {
                const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
                multiply_r_columns<<<grid, block, 0, context.stream>>>(workspace.r.get(), workspace.norms.get(), workspace.n);
                KERNEL_CHECK();
            }
            result.factorization_success = true;
            return;
        }

        // Iterative Triangular Polish loop (quadratic defect reduction per step)
        double current = result.initial_orthogonality_fro;
        const dim3 block(16, 16), grid((workspace.n + 15) / 16, (workspace.n + 15) / 16);
        const double one = 1.0, zero = 0.0;

        while (current > config.orthogonality_tolerance && result.refinements < config.maximum_refinements) {
            // Form upper triangular P = I - T in workspace.polar_h from upper triangle of gram
            result.time.ns_small_gemm_ms += timer.measure([&] {
                form_triangular_polish_matrix<<<grid, block, 0, context.stream>>>(
                    workspace.gram.get(), workspace.polar_h.get(), workspace.n);
                KERNEL_CHECK();
            });

            // Update Q+ = Q * P using high-throughput cublasDgemm
            result.time.ns_tall_gemm_ms += timer.measure([&] {
                CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                         workspace.m, workspace.n, workspace.n,
                                         &one, workspace.q.get(), workspace.m,
                                         workspace.polar_h.get(), workspace.n,
                                         &zero, workspace.temporary_q.get(), workspace.m));
            });
            std::swap(workspace.q, workspace.temporary_q);

            // Update R+ = P^{-1} R in-place using cublasDtrsm
            if (need_r) {
                result.time.r_update_ms += timer.measure([&] {
                    CUBLAS_CHECK(cublasDtrsm(context.blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                             CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                             workspace.n, workspace.n,
                                             &one, workspace.polar_h.get(), workspace.n,
                                             workspace.r.get(), workspace.n));
                });
            }

            result.passes++;
            result.refinements++;

            result.time.stopping_check_ms += timer.measure([&] {
                form_fp64_gram(workspace, context, workspace.q.get());
                current = orthogonality_from_gram(workspace, context);
            });
        }

        if (need_r && config.equilibrate_columns) {
            multiply_r_columns<<<grid, block, 0, context.stream>>>(workspace.r.get(), workspace.norms.get(), workspace.n);
            KERNEL_CHECK();
        }

        result.accuracy_target_met = std::isfinite(current) && current <= config.orthogonality_tolerance;
        result.stop_reason = result.accuracy_target_met ? StopReason::ToleranceMet : StopReason::PassLimit;
        result.factorization_success = true;
    });
    return result;
}

static RunResult execute(Workspace& workspace, LibraryContext& context, const RunConfig& config) {
    RunResult result;
    switch (config.algorithm) {
        case Algorithm::CholeskyQR2: result = run_choleskyqr2(workspace, context, config); break;
        case Algorithm::ShiftedCholeskyQR: result = run_shifted_choleskyqr(workspace, context, config); break;
        case Algorithm::ShiftedCholeskyQR3: result = run_shifted_choleskyqr3(workspace, context, config); break;
        case Algorithm::AdaptiveFP64: result = run_adaptive(workspace, context, config, 0); break;
        case Algorithm::Int8L1Adaptive: result = run_adaptive(workspace, context, config, 1); break;
        case Algorithm::Int8L2Adaptive: result = run_adaptive(workspace, context, config, 2); break;
        case Algorithm::Int8L1CholeskyQR: result = run_int8_choleskyqr(workspace, context, config, 1); break;
        case Algorithm::Int8L2CholeskyQR: result = run_int8_choleskyqr(workspace, context, config, 2); break;
        case Algorithm::Int8L2PolarNS: result = run_int8_l2_polar_ns(workspace, context, config); break;
        case Algorithm::Int8L2PolarNSQR: result = run_int8_l2_polar_ns(workspace, context, config); break;
        case Algorithm::Int8L2PolarChebyshev: result = run_int8_l2_polar_chebyshev(workspace, context, config); break;
        case Algorithm::Int8L2PolarChebyshevQR: result = run_int8_l2_polar_chebyshev(workspace, context, config); break;
        case Algorithm::Int8L2NIMARefinement: result = run_int8_l2_nima_refinement(workspace, context, config); break;
        case Algorithm::Int8L2NIMARefinementQR: result = run_int8_l2_nima_refinement(workspace, context, config); break;
        case Algorithm::Int8L2ShiftRetry: result = run_int8_l2_shift_retry(workspace, context, config); break;
        case Algorithm::Int8L2ShiftRetryQR: result = run_int8_l2_shift_retry(workspace, context, config); break;
        case Algorithm::HouseholderQR: result = run_householder(workspace, context, config); break;
        case Algorithm::Int8L2CertifiedRidge: result = run_int8_l2_certified_ridge(workspace, context, config); break;
        case Algorithm::Int8L2CertifiedRidgeQR: result = run_int8_l2_certified_ridge(workspace, context, config); break;
        case Algorithm::Int8L2TriangularPolish: result = run_int8_l2_triangular_polish(workspace, context, config); break;
        case Algorithm::Int8L2TriangularPolishQR: result = run_int8_l2_triangular_polish(workspace, context, config); break;
    }
    if (!result.factorization_success) return result;
    result.final_orthogonality_fro = compute_orthogonality(workspace, context);
    if (config.algorithm != Algorithm::AdaptiveFP64 && config.algorithm != Algorithm::Int8L1Adaptive &&
        config.algorithm != Algorithm::Int8L2Adaptive && config.algorithm != Algorithm::Int8L2ShiftRetry &&
        config.algorithm != Algorithm::Int8L2ShiftRetryQR &&
        config.algorithm != Algorithm::Int8L2CertifiedRidge && config.algorithm != Algorithm::Int8L2CertifiedRidgeQR &&
        config.algorithm != Algorithm::Int8L2TriangularPolish && config.algorithm != Algorithm::Int8L2TriangularPolishQR)
        result.accuracy_target_met = result.final_orthogonality_fro <= config.orthogonality_tolerance;
    if (config.output == OutputContract::QR) {
        double norm = 0.0;
        CUBLAS_CHECK(cublasDnrm2(context.blas, workspace.m * workspace.n, workspace.a.get(), 1, &norm));
        result.relative_residual_fro = compute_residual(workspace, context, norm);
    }
    return result;
}

static void generate_synthetic(Workspace& workspace, LibraryContext& context, double condition, std::uint64_t seed) {
    const std::size_t u_elements = static_cast<std::size_t>(workspace.m) * workspace.n;
    const std::size_t v_elements = static_cast<std::size_t>(workspace.n) * workspace.n;
    DeviceBuffer<double> u((u_elements + 1) & ~std::size_t{1});
    DeviceBuffer<double> v((v_elements + 1) & ~std::size_t{1});
    DeviceBuffer<double> local_tau(workspace.n), device_sigma(workspace.n);
    DeviceBuffer<int> local_info(1);
    int qr_mn = 0, qr_nn = 0, q_mn = 0, q_nn = 0;
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(context.solver, workspace.m, workspace.n, u.get(), workspace.m, &qr_mn));
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(context.solver, workspace.n, workspace.n, v.get(), workspace.n, &qr_nn));
    CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(context.solver, workspace.m, workspace.n, workspace.n, u.get(), workspace.m, local_tau.get(), &q_mn));
    CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(context.solver, workspace.n, workspace.n, workspace.n, v.get(), workspace.n, local_tau.get(), &q_nn));
    DeviceBuffer<double> local_work(std::max({qr_mn, qr_nn, q_mn, q_nn}));
    curandGenerator_t random = nullptr;
    CURAND_CHECK(curandCreateGenerator(&random, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetStream(random, context.stream));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(random, seed));
    CURAND_CHECK(curandGenerateNormalDouble(random, u.get(), u.size(), 0.0, 1.0));
    CURAND_CHECK(curandGenerateNormalDouble(random, v.get(), v.size(), 0.0, 1.0));
    curandDestroyGenerator(random);
    CUSOLVER_CHECK(cusolverDnDgeqrf(context.solver, workspace.m, workspace.n, u.get(), workspace.m,
                                    local_tau.get(), local_work.get(), local_work.size(), local_info.get()));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));
    int factor_info = 0;
    CUDA_CHECK(cudaMemcpy(&factor_info, local_info.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (factor_info != 0) throw std::runtime_error("synthetic U geqrf failed with info=" + std::to_string(factor_info));
    CUSOLVER_CHECK(cusolverDnDorgqr(context.solver, workspace.m, workspace.n, workspace.n, u.get(), workspace.m,
                                    local_tau.get(), local_work.get(), local_work.size(), local_info.get()));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));
    CUDA_CHECK(cudaMemcpy(&factor_info, local_info.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (factor_info != 0) throw std::runtime_error("synthetic U orgqr failed with info=" + std::to_string(factor_info));
    CUSOLVER_CHECK(cusolverDnDgeqrf(context.solver, workspace.n, workspace.n, v.get(), workspace.n,
                                    local_tau.get(), local_work.get(), local_work.size(), local_info.get()));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));
    CUDA_CHECK(cudaMemcpy(&factor_info, local_info.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (factor_info != 0) throw std::runtime_error("synthetic V geqrf failed with info=" + std::to_string(factor_info));
    CUSOLVER_CHECK(cusolverDnDorgqr(context.solver, workspace.n, workspace.n, workspace.n, v.get(), workspace.n,
                                    local_tau.get(), local_work.get(), local_work.size(), local_info.get()));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));
    CUDA_CHECK(cudaMemcpy(&factor_info, local_info.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (factor_info != 0) throw std::runtime_error("synthetic V orgqr failed with info=" + std::to_string(factor_info));
    std::vector<double> host_sigma(workspace.n);
    for (int index = 0; index < workspace.n; ++index) {
        const double position = workspace.n == 1 ? 0.0 : static_cast<double>(index) / (workspace.n - 1);
        host_sigma[index] = std::pow(1.0 / condition, position);
    }
    CUDA_CHECK(cudaMemcpyAsync(device_sigma.get(), host_sigma.data(), workspace.n * sizeof(double),
                               cudaMemcpyHostToDevice, context.stream));
    scale_columns<<<dim3(workspace.n, (workspace.m + 255) / 256), 256, 0, context.stream>>>(
        u.get(), device_sigma.get(), workspace.m, workspace.n);
    KERNEL_CHECK();
    const double one = 1.0, zero = 0.0;
    CUBLAS_CHECK(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_T, workspace.m, workspace.n, workspace.n,
                             &one, u.get(), workspace.m, v.get(), workspace.n, &zero, workspace.a.get(), workspace.m));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));
}

static std::vector<double> load_qr8(const std::string& path, std::uint64_t& rows, std::uint64_t& columns) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path);
    std::array<char, 8> magic{};
    stream.read(magic.data(), magic.size());
    if (std::string(magic.data(), magic.size()) != "QR8MAT01") throw std::runtime_error("invalid QR8 matrix magic");
    stream.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    stream.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    std::vector<double> values(rows * columns);
    stream.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(double));
    if (!stream) throw std::runtime_error("truncated QR8 matrix");
    return values;
}

static bool algorithm_selected(Algorithm algorithm, const std::string& token) {
    if (token == "all") return true;
    if (token == "fp64") {
        return algorithm == Algorithm::CholeskyQR2 || algorithm == Algorithm::ShiftedCholeskyQR ||
               algorithm == Algorithm::ShiftedCholeskyQR3 || algorithm == Algorithm::AdaptiveFP64;
    }
    if (token == "chol") return algorithm != Algorithm::HouseholderQR;
    if (token == "adaptive") {
        return algorithm == Algorithm::AdaptiveFP64 || algorithm == Algorithm::Int8L1Adaptive ||
               algorithm == Algorithm::Int8L2Adaptive;
    }
    if (token == "int8") {
        return algorithm == Algorithm::Int8L1Adaptive || algorithm == Algorithm::Int8L2Adaptive ||
               algorithm == Algorithm::Int8L1CholeskyQR || algorithm == Algorithm::Int8L2CholeskyQR ||
               algorithm == Algorithm::Int8L2PolarNS || algorithm == Algorithm::Int8L2PolarNSQR ||
               algorithm == Algorithm::Int8L2PolarChebyshev || algorithm == Algorithm::Int8L2PolarChebyshevQR ||
               algorithm == Algorithm::Int8L2NIMARefinement || algorithm == Algorithm::Int8L2NIMARefinementQR ||
               algorithm == Algorithm::Int8L2ShiftRetry || algorithm == Algorithm::Int8L2ShiftRetryQR ||
               algorithm == Algorithm::Int8L2CertifiedRidge || algorithm == Algorithm::Int8L2CertifiedRidgeQR ||
               algorithm == Algorithm::Int8L2TriangularPolish || algorithm == Algorithm::Int8L2TriangularPolishQR;
    }
    if (token == "int8chol") {
        return algorithm == Algorithm::Int8L1Adaptive || algorithm == Algorithm::Int8L2Adaptive ||
               algorithm == Algorithm::Int8L1CholeskyQR || algorithm == Algorithm::Int8L2CholeskyQR;
    }
    if (token == "fixedchol") {
        return algorithm == Algorithm::Int8L1CholeskyQR || algorithm == Algorithm::Int8L2CholeskyQR;
    }
    if (token == "ns" || token == "polarns")
        return algorithm == Algorithm::Int8L2PolarNS || algorithm == Algorithm::Int8L2PolarNSQR;
    if (token == "polarnsqr") return algorithm == Algorithm::Int8L2PolarNSQR;
    if (token == "cheb" || token == "polarcheb")
        return algorithm == Algorithm::Int8L2PolarChebyshev || algorithm == Algorithm::Int8L2PolarChebyshevQR;
    if (token == "polarchebqr") return algorithm == Algorithm::Int8L2PolarChebyshevQR;
    if (token == "nima")
        return algorithm == Algorithm::Int8L2NIMARefinement || algorithm == Algorithm::Int8L2NIMARefinementQR;
    if (token == "nimaqr") return algorithm == Algorithm::Int8L2NIMARefinementQR;
    if (token == "householder") return algorithm == Algorithm::HouseholderQR;
    if (token == "baselines")
        return algorithm != Algorithm::Int8L2PolarNS && algorithm != Algorithm::Int8L2PolarNSQR &&
               algorithm != Algorithm::Int8L2PolarChebyshev && algorithm != Algorithm::Int8L2PolarChebyshevQR &&
               algorithm != Algorithm::Int8L2NIMARefinement && algorithm != Algorithm::Int8L2NIMARefinementQR &&
               algorithm != Algorithm::Int8L2ShiftRetry && algorithm != Algorithm::Int8L2ShiftRetryQR;
    if (token == "cholqr2") return algorithm == Algorithm::CholeskyQR2;
    if (token == "shifted") return algorithm == Algorithm::ShiftedCholeskyQR;
    if (token == "shifted3") return algorithm == Algorithm::ShiftedCholeskyQR3;
    if (token == "fp64adaptive") return algorithm == Algorithm::AdaptiveFP64;
    if (token == "int8l1") return algorithm == Algorithm::Int8L1Adaptive;
    if (token == "int8l2") return algorithm == Algorithm::Int8L2Adaptive;
    if (token == "int8l1chol") return algorithm == Algorithm::Int8L1CholeskyQR;
    if (token == "int8l2chol") return algorithm == Algorithm::Int8L2CholeskyQR;
    if (token == "shiftretry") return algorithm == Algorithm::Int8L2ShiftRetry || algorithm == Algorithm::Int8L2ShiftRetryQR;
    if (token == "shiftretryqr") return algorithm == Algorithm::Int8L2ShiftRetryQR;
    if (token == "ridge" || token == "certifiedridge")
        return algorithm == Algorithm::Int8L2CertifiedRidge || algorithm == Algorithm::Int8L2CertifiedRidgeQR;
    if (token == "ridgeqr" || token == "certifiedridgeqr")
        return algorithm == Algorithm::Int8L2CertifiedRidgeQR;
    if (token == "triangular" || token == "polish" || token == "triangularpolish")
        return algorithm == Algorithm::Int8L2TriangularPolish || algorithm == Algorithm::Int8L2TriangularPolishQR;
    if (token == "triangularqr" || token == "polishqr" || token == "triangularpolishqr")
        return algorithm == Algorithm::Int8L2TriangularPolishQR;
    return false;
}

static std::vector<std::string> split_tokens(const std::string& text) {
    std::vector<std::string> tokens;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        const auto first = token.find_first_not_of(" \t\r\n");
        const auto last = token.find_last_not_of(" \t\r\n");
        if (first != std::string::npos) tokens.push_back(token.substr(first, last - first + 1));
    }
    return tokens;
}

static std::vector<Algorithm> select_algorithms(const std::string& specification, OutputContract output) {
    const std::vector<Algorithm> candidates = {
        Algorithm::CholeskyQR2, Algorithm::ShiftedCholeskyQR, Algorithm::ShiftedCholeskyQR3,
        Algorithm::AdaptiveFP64, Algorithm::Int8L1Adaptive, Algorithm::Int8L2Adaptive,
        Algorithm::Int8L1CholeskyQR, Algorithm::Int8L2CholeskyQR,
        Algorithm::Int8L2PolarNS, Algorithm::Int8L2PolarNSQR,
        Algorithm::Int8L2PolarChebyshev, Algorithm::Int8L2PolarChebyshevQR,
        Algorithm::Int8L2NIMARefinement, Algorithm::Int8L2NIMARefinementQR,
        Algorithm::Int8L2ShiftRetry, Algorithm::Int8L2ShiftRetryQR,
        Algorithm::HouseholderQR,
        Algorithm::Int8L2CertifiedRidge, Algorithm::Int8L2CertifiedRidgeQR,
        Algorithm::Int8L2TriangularPolish, Algorithm::Int8L2TriangularPolishQR
    };
    std::vector<Algorithm> selected;
    for (const std::string& token : split_tokens(specification)) {
        bool recognized = false;
        for (Algorithm algorithm : candidates) {
            if (!algorithm_selected(algorithm, token)) continue;
            recognized = true;
            if ((algorithm == Algorithm::Int8L2PolarNS ||
                 algorithm == Algorithm::Int8L2PolarChebyshev ||
                 algorithm == Algorithm::Int8L2NIMARefinement ||
                 algorithm == Algorithm::Int8L2ShiftRetry ||
                 algorithm == Algorithm::Int8L2CertifiedRidge ||
                 algorithm == Algorithm::Int8L2TriangularPolish) &&
                output != OutputContract::QOnly) continue;
            if ((algorithm == Algorithm::Int8L2PolarNSQR ||
                 algorithm == Algorithm::Int8L2PolarChebyshevQR ||
                 algorithm == Algorithm::Int8L2NIMARefinementQR ||
                 algorithm == Algorithm::Int8L2ShiftRetryQR ||
                 algorithm == Algorithm::Int8L2CertifiedRidgeQR ||
                 algorithm == Algorithm::Int8L2TriangularPolishQR) &&
                output != OutputContract::QR) continue;
            if (std::find(selected.begin(), selected.end(), algorithm) == selected.end())
                selected.push_back(algorithm);
        }
        if (!recognized) throw std::runtime_error("unknown method selector: " + token);
    }
    if (selected.empty()) throw std::runtime_error("method selection is empty for this output contract");
    return selected;
}

struct CommandLine {
    int m = 4096, n = 256, warmups = 3, trials = 20, maximum_refinements = 1;
    int maximum_inverse_sqrt_iterations = 12, maximum_polar_iterations = 8;
    int maximum_chebyshev_degree = 128;
    double condition = 1.0, tolerance = 1.0e-12;
    double inverse_sqrt_tolerance = 1.0e-10;
    double l2_shift = 0.0;
    int l2_shift_retries = 1;
    std::uint64_t seed = 1;
    bool equilibrate = false;
    bool kernel_timing = false;
    OutputContract output = OutputContract::QR;
    std::string input, matrix_id = "synthetic", raw_csv = "results/raw.csv", methods = "all";
};

static CommandLine parse_arguments(int argc, char** argv) {
    CommandLine command;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc) throw std::runtime_error("missing value after " + argument);
            return argv[index];
        };
        if (argument == "--m") command.m = std::stoi(value());
        else if (argument == "--n") command.n = std::stoi(value());
        else if (argument == "--condition") command.condition = std::stod(value());
        else if (argument == "--seed") command.seed = std::stoull(value());
        else if (argument == "--warmups") command.warmups = std::stoi(value());
        else if (argument == "--trials") command.trials = std::stoi(value());
        else if (argument == "--tolerance") command.tolerance = std::stod(value());
        else if (argument == "--maximum-refinements") command.maximum_refinements = std::stoi(value());
        else if (argument == "--ns-inverse-iterations") command.maximum_inverse_sqrt_iterations = std::stoi(value());
        else if (argument == "--ns-polar-iterations") command.maximum_polar_iterations = std::stoi(value());
        else if (argument == "--ns-inverse-tolerance") command.inverse_sqrt_tolerance = std::stod(value());
        else if (argument == "--chebyshev-max-degree") command.maximum_chebyshev_degree = std::stoi(value());
        else if (argument == "--l2-shift") command.l2_shift = std::stod(value());
        else if (argument == "--l2-shift-retries") command.l2_shift_retries = std::stoi(value());
        else if (argument == "--input") command.input = value();
        else if (argument == "--matrix-id") command.matrix_id = value();
        else if (argument == "--raw-csv") command.raw_csv = value();
        else if (argument == "--methods") command.methods = value();
        else if (argument == "--equilibrate") command.equilibrate = true;
        else if (argument == "--kernel-timing") command.kernel_timing = true;
        else if (argument == "--output") {
            const std::string contract = value();
            if (contract == "q") command.output = OutputContract::QOnly;
            else if (contract == "qr") command.output = OutputContract::QR;
            else throw std::runtime_error("--output must be q or qr");
        }
        else if (argument == "--help") {
            std::cout << "qr8_benchmark [--m M --n N --condition K --seed S | --input matrix.qr8] "
                         "[--warmups W --trials T --output q|qr --tolerance E --maximum-refinements R "
                         "--ns-inverse-iterations I --ns-polar-iterations P --ns-inverse-tolerance E "
                         "--chebyshev-max-degree D "
                         "--methods all|fp64|adaptive|int8|int8chol|fixedchol|ns|cheb|nima|householder[,token...] "
                         "--equilibrate --kernel-timing --raw-csv FILE]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument " + argument);
    }
    if (command.m < command.n || command.n <= 0 || command.trials < 1 || command.warmups < 0 ||
        command.condition < 1.0 || command.tolerance <= 0.0 || command.maximum_refinements < 0 ||
        command.maximum_inverse_sqrt_iterations < 0 || command.maximum_polar_iterations < 0 ||
        command.maximum_chebyshev_degree < 1 ||
        command.inverse_sqrt_tolerance <= 0.0)
        throw std::runtime_error("invalid dimensions or trial counts");
    return command;
}

}  // namespace qr8

int main(int argc, char** argv) {
    try {
        using namespace qr8;
        CommandLine command = parse_arguments(argc, argv);
        LibraryContext context;
        std::vector<double> host_input;
        if (!command.input.empty()) {
            std::uint64_t rows = 0, columns = 0;
            host_input = load_qr8(command.input, rows, columns);
            if (rows > INT_MAX || columns > INT_MAX) throw std::runtime_error("matrix dimensions exceed CUDA integer API limits");
            command.m = static_cast<int>(rows); command.n = static_cast<int>(columns);
        }
        Workspace workspace(command.m, command.n, context);
        if (command.input.empty()) generate_synthetic(workspace, context, command.condition, command.seed);
        else CUDA_CHECK(cudaMemcpyAsync(workspace.a.get(), host_input.data(), host_input.size() * sizeof(double),
                                        cudaMemcpyHostToDevice, context.stream));
        CUDA_CHECK(cudaStreamSynchronize(context.stream));

        std::ofstream raw(command.raw_csv);
        if (!raw) throw std::runtime_error("cannot open raw CSV " + command.raw_csv);
        raw << std::setprecision(17);
        raw << "matrix_id,source_kind,requested_condition,seed,m,n,algorithm,output,equilibrated,timing_mode,trial,factorization_success,target_met,potrf_info,passes,refinements,inverse_sqrt_iterations,polar_iterations,chebyshev_degree,stop_reason,shift,initial_orth_fro,final_orth_fro,residual_fro,inverse_sqrt_residual_fro,quantization_residual_fro,total_ms,preprocess_ms,gram_ms,quantization_ms,cholesky_ms,trsm_ms,r_update_ms,stopping_check_ms,householder_factorization_ms,explicit_q_ms,shift_ms,ns_small_gemm_ms,ns_tall_gemm_ms,small_qr_ms\n";

        std::vector<Algorithm> algorithms = select_algorithms(command.methods, command.output);
        std::array<std::vector<RunResult>, 32> results_by_algorithm;
        for (Algorithm algorithm : algorithms) {
            RunConfig config;
            config.algorithm = algorithm;
            config.output = command.output;
            config.equilibrate_columns = command.equilibrate;
            config.orthogonality_tolerance = command.tolerance;
            config.maximum_refinements = command.maximum_refinements;
            config.collect_phase_timings = command.kernel_timing;
            config.maximum_inverse_sqrt_iterations = command.maximum_inverse_sqrt_iterations;
            config.maximum_polar_iterations = command.maximum_polar_iterations;
            config.maximum_chebyshev_degree = command.maximum_chebyshev_degree;
            config.inverse_sqrt_tolerance = command.inverse_sqrt_tolerance;
            config.l2_shift = command.l2_shift;
            config.l2_shift_retries = command.l2_shift_retries;
            for (int warmup = 0; warmup < command.warmups; ++warmup) (void)execute(workspace, context, config);
        }
        std::mt19937_64 order_generator(command.seed ^ 0x9e3779b97f4a7c15ULL);
        for (int trial = 0; trial < command.trials; ++trial) {
            std::vector<Algorithm> trial_order = algorithms;
            std::shuffle(trial_order.begin(), trial_order.end(), order_generator);
            for (Algorithm algorithm : trial_order) {
                RunConfig config;
                config.algorithm = algorithm;
                config.output = command.output;
                config.equilibrate_columns = command.equilibrate;
                config.orthogonality_tolerance = command.tolerance;
                config.maximum_refinements = command.maximum_refinements;
                config.collect_phase_timings = command.kernel_timing;
                config.maximum_inverse_sqrt_iterations = command.maximum_inverse_sqrt_iterations;
                config.maximum_polar_iterations = command.maximum_polar_iterations;
                config.inverse_sqrt_tolerance = command.inverse_sqrt_tolerance;
                config.l2_shift = command.l2_shift;
                config.l2_shift_retries = command.l2_shift_retries;
                RunResult result = execute(workspace, context, config);
                results_by_algorithm[static_cast<std::size_t>(algorithm)].push_back(result);
                raw << command.matrix_id << ',' << (command.input.empty() ? "synthetic" : "real") << ','
                    << (command.input.empty() ? command.condition : NAN) << ',' << command.seed << ','
                    << command.m << ',' << command.n << ',' << algorithm_name(algorithm) << ','
                    << output_name(config.output) << ',' << (command.equilibrate ? 1 : 0) << ','
                    << (command.kernel_timing ? "instrumented" : "primary") << ',' << trial << ','
                    << (result.factorization_success ? 1 : 0) << ',' << (result.accuracy_target_met ? 1 : 0) << ','
                    << result.potrf_info << ',' << result.passes << ',' << result.refinements << ','
                    << result.inverse_sqrt_iterations << ',' << result.polar_iterations << ','
                    << result.chebyshev_degree << ','
                    << stop_reason_name(result.stop_reason) << ',' << result.shift << ',' << result.initial_orthogonality_fro << ','
                    << result.final_orthogonality_fro << ',' << result.relative_residual_fro << ','
                    << result.inverse_sqrt_residual_fro << ',' << result.quantization_residual_fro << ','
                    << result.time.total_ms << ','
                    << result.time.preprocessing_ms << ',' << result.time.gram_ms << ',' << result.time.quantization_ms << ','
                    << result.time.cholesky_ms << ',' << result.time.triangular_solve_ms << ',' << result.time.r_update_ms << ','
                    << result.time.stopping_check_ms << ',' << result.time.householder_factorization_ms << ','
                    << result.time.explicit_q_ms << ',' << result.time.shift_ms << ','
                    << result.time.ns_small_gemm_ms << ',' << result.time.ns_tall_gemm_ms << ','
                    << result.time.small_qr_ms << '\n';
            }
        }
        for (Algorithm algorithm : algorithms) {
            const TrialStatistics stats = summarize_times(results_by_algorithm[static_cast<std::size_t>(algorithm)]);
            std::cout << algorithm_name(algorithm) << ": successes=" << stats.count << '/' << command.trials
                      << " mean=" << stats.mean_ms << " ms stddev=" << stats.standard_deviation_ms
                      << " median=" << stats.median_ms << " min=" << stats.minimum_ms << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
