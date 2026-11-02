#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace qr8 {

enum class Algorithm {
    CholeskyQR2,
    ShiftedCholeskyQR,
    ShiftedCholeskyQR3,
    AdaptiveFP64,
    Int8L1Adaptive,
    Int8L2Adaptive,
    Int8L1CholeskyQR,
    Int8L2CholeskyQR,
    Int8L2PolarNS,
    Int8L2PolarNSQR,
    Int8L2PolarChebyshev,
    Int8L2PolarChebyshevQR,
    Int8L2NIMARefinement,
    Int8L2NIMARefinementQR,
    Int8L2ShiftRetry,
    Int8L2ShiftRetryQR,
    HouseholderQR,
    Int8L2CertifiedRidge,
    Int8L2CertifiedRidgeQR,
    Int8L2TriangularPolish,
    Int8L2TriangularPolishQR
};

enum class OutputContract { QOnly, QR };
enum class StopReason {
    FixedPasses,
    ToleranceMet,
    PassLimit,
    PotrfFailure,
    InverseSqrtLimit,
    PolarLimit,
    ChebyshevDegreeLimit,
    InvalidInput
};

struct RunConfig {
    Algorithm algorithm = Algorithm::CholeskyQR2;
    OutputContract output = OutputContract::QR;
    double orthogonality_tolerance = 1.0e-12;
    int maximum_refinements = 1;
    bool equilibrate_columns = false;
    bool collect_phase_timings = false;
    int maximum_inverse_sqrt_iterations = 12;
    int maximum_polar_iterations = 8;
    int maximum_chebyshev_degree = 128;
    double inverse_sqrt_tolerance = 1.0e-10;
    double l2_shift = 0.0;
    int l2_shift_retries = 1;
};

struct PhaseTimes {
    double total_ms = 0.0;
    double preprocessing_ms = 0.0;
    double gram_ms = 0.0;
    double quantization_ms = 0.0;
    double cholesky_ms = 0.0;
    double triangular_solve_ms = 0.0;
    double r_update_ms = 0.0;
    double stopping_check_ms = 0.0;
    double householder_factorization_ms = 0.0;
    double explicit_q_ms = 0.0;
    double shift_ms = 0.0;
    double ns_small_gemm_ms = 0.0;
    double ns_tall_gemm_ms = 0.0;
    double small_qr_ms = 0.0;
};

struct RunResult {
    bool factorization_success = false;
    bool accuracy_target_met = false;
    int potrf_info = 0;
    int passes = 0;
    int refinements = 0;
    int inverse_sqrt_iterations = 0;
    int polar_iterations = 0;
    int chebyshev_degree = 0;
    StopReason stop_reason = StopReason::InvalidInput;
    double shift = 0.0;
    double initial_orthogonality_fro = NAN;
    double final_orthogonality_fro = NAN;
    double relative_residual_fro = NAN;
    double inverse_sqrt_residual_fro = NAN;
    double quantization_residual_fro = NAN;
    PhaseTimes time;
};

struct TrialStatistics {
    std::size_t count = 0;
    double mean_ms = NAN;
    double standard_deviation_ms = NAN;
    double minimum_ms = NAN;
    double median_ms = NAN;
};

const char* algorithm_name(Algorithm algorithm);
const char* output_name(OutputContract output);
const char* stop_reason_name(StopReason reason);
TrialStatistics summarize_times(const std::vector<RunResult>& results);

}  // namespace qr8
