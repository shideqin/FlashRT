#include "omnivoice_bf16_matmul.cuh"

#include <cublasLt.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace flash_rt::kernels {
namespace {

#define FLASHRT_OMNIVOICE_CUBLASLT_CHECK(expr)                         \
    do {                                                                \
        const cublasStatus_t _status = (expr);                           \
        if (_status != CUBLAS_STATUS_SUCCESS) {                          \
            throw std::runtime_error(                                   \
                std::string("OmniVoice cuBLASLt error ") +             \
                std::to_string(static_cast<int>(_status)) +              \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));  \
        }                                                               \
    } while (0)

constexpr size_t kWorkspaceBytes = 32 * 1024 * 1024;

struct PlanKey {
    int m;
    int n;
    int k;
    int reference_m;
    int algorithm_rank;

    bool operator==(const PlanKey& other) const {
        return m == other.m && n == other.n && k == other.k &&
               reference_m == other.reference_m &&
               algorithm_rank == other.algorithm_rank;
    }
};

struct PlanKeyHash {
    size_t operator()(const PlanKey& key) const {
        size_t hash = static_cast<size_t>(key.m);
        hash = hash * 1315423911u + static_cast<size_t>(key.n);
        hash = hash * 1315423911u + static_cast<size_t>(key.k);
        hash = hash * 1315423911u + static_cast<size_t>(key.reference_m);
        hash = hash * 1315423911u + static_cast<size_t>(key.algorithm_rank);
        return hash;
    }
};

struct MatmulPlan {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t x_layout = nullptr;
    cublasLtMatrixLayout_t weight_layout = nullptr;
    cublasLtMatrixLayout_t out_layout = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
};

cublasLtHandle_t g_handle = nullptr;
void* g_workspace = nullptr;
std::mutex g_plan_mutex;
std::unordered_map<PlanKey, MatmulPlan, PlanKeyHash> g_plans;

void ensure_handle() {
    if (g_handle != nullptr) {
        return;
    }
    FLASHRT_OMNIVOICE_CUBLASLT_CHECK(cublasLtCreate(&g_handle));
    const cudaError_t status = cudaMalloc(&g_workspace, kWorkspaceBytes);
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("OmniVoice cuBLASLt workspace allocation failed: ") +
            cudaGetErrorString(status));
    }
}

void configure_operation(cublasLtMatmulDesc_t* operation) {
    const cublasOperation_t op_n = CUBLAS_OP_N;
    const cublasOperation_t op_t = CUBLAS_OP_T;
    FLASHRT_OMNIVOICE_CUBLASLT_CHECK(cublasLtMatmulDescCreate(
        operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    FLASHRT_OMNIVOICE_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        *operation, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));
    FLASHRT_OMNIVOICE_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        *operation, CUBLASLT_MATMUL_DESC_TRANSB, &op_t, sizeof(op_t)));
}

void configure_layout(
    cublasLtMatrixLayout_t* layout,
    int64_t rows,
    int64_t cols,
    int64_t leading_dimension) {
    const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    FLASHRT_OMNIVOICE_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
        layout, CUDA_R_16BF, rows, cols, leading_dimension));
    FLASHRT_OMNIVOICE_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        *layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &row_order, sizeof(row_order)));
}

cublasLtMatmulAlgo_t select_reference_algorithm(
    int reference_m, int n, int k, int algorithm_rank) {
    constexpr int kMaxHeuristicRanks = 32;
    if (algorithm_rank < 0 || algorithm_rank >= kMaxHeuristicRanks) {
        throw std::invalid_argument(
            "OmniVoice cuBLASLt heuristic rank is out of range");
    }
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t x_layout = nullptr;
    cublasLtMatrixLayout_t weight_layout = nullptr;
    cublasLtMatrixLayout_t out_layout = nullptr;
    configure_operation(&operation);
    configure_layout(&x_layout, reference_m, k, k);
    configure_layout(&weight_layout, n, k, k);
    configure_layout(&out_layout, reference_m, n, n);

    cublasLtMatmulPreference_t preference = nullptr;
    FLASHRT_OMNIVOICE_CUBLASLT_CHECK(
        cublasLtMatmulPreferenceCreate(&preference));
    FLASHRT_OMNIVOICE_CUBLASLT_CHECK(
        cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &kWorkspaceBytes, sizeof(kWorkspaceBytes)));
    cublasLtMatmulHeuristicResult_t heuristics[kMaxHeuristicRanks]{};
    int returned = 0;
    FLASHRT_OMNIVOICE_CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        g_handle, operation, x_layout, weight_layout,
        out_layout, out_layout, preference, algorithm_rank + 1,
        heuristics, &returned));

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(x_layout);
    cublasLtMatrixLayoutDestroy(weight_layout);
    cublasLtMatrixLayoutDestroy(out_layout);
    cublasLtMatmulDescDestroy(operation);
    if (returned <= algorithm_rank) {
        throw std::runtime_error(
            "OmniVoice cuBLASLt heuristic rank is unavailable");
    }
    return heuristics[algorithm_rank].algo;
}

MatmulPlan& get_plan(
    int m, int n, int k, int reference_m, int algorithm_rank) {
    const PlanKey key{m, n, k, reference_m, algorithm_rank};
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    ensure_handle();
    const auto found = g_plans.find(key);
    if (found != g_plans.end()) {
        return found->second;
    }

    MatmulPlan plan;
    configure_operation(&plan.operation);
    configure_layout(&plan.x_layout, m, k, k);
    configure_layout(&plan.weight_layout, n, k, k);
    configure_layout(&plan.out_layout, m, n, n);
    plan.algorithm = select_reference_algorithm(
        reference_m, n, k, algorithm_rank);
    return g_plans.emplace(key, plan).first->second;
}

}  // namespace

void omnivoice_bf16_matmul_reference_algo(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    int M,
    int N,
    int K,
    int reference_M,
    cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0 || reference_M < M) {
        throw std::invalid_argument(
            "OmniVoice BF16 reference GEMM received an invalid shape");
    }
    MatmulPlan& plan = get_plan(M, N, K, reference_M, 0);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    FLASHRT_OMNIVOICE_CUBLASLT_CHECK(cublasLtMatmul(
        g_handle, plan.operation, &alpha,
        x, plan.x_layout,
        weight, plan.weight_layout,
        &beta,
        out, plan.out_layout,
        out, plan.out_layout,
        &plan.algorithm,
        g_workspace, kWorkspaceBytes,
        stream));
}

void omnivoice_bf16_matmul_heuristic_algo(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    int M,
    int N,
    int K,
    int reference_M,
    int algorithm_rank,
    cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0 || reference_M < M ||
        algorithm_rank < 0) {
        throw std::invalid_argument(
            "OmniVoice BF16 heuristic GEMM received invalid arguments");
    }
    MatmulPlan& plan = get_plan(
        M, N, K, reference_M, algorithm_rank);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    FLASHRT_OMNIVOICE_CUBLASLT_CHECK(cublasLtMatmul(
        g_handle, plan.operation, &alpha,
        x, plan.x_layout,
        weight, plan.weight_layout,
        &beta,
        out, plan.out_layout,
        out, plan.out_layout,
        &plan.algorithm,
        g_workspace, kWorkspaceBytes,
        stream));
}

}  // namespace flash_rt::kernels
