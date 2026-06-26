// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS NVFP4 W4A16 GEMM with fused residual add epilogue, SM120a.
// Implementation. See header for API docs.
//
// Reuses the existing NVFP4 GEMM templates from
// cutlass_nvfp4_w4a16_gemm_sm120.cu — only changes the epilogue
// arguments to pass residual_bf16 as C with beta=1.0.

#include "cutlass_nvfp4_gemm_residual_sm120.cuh"

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"

#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace flash_rt {
namespace gemm {

namespace {

using namespace cute;

using ElementA           = cutlass::float_e2m1_t;
using ElementB           = cutlass::float_e2m1_t;
using ElementC           = cutlass::bfloat16_t;
using ElementD           = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute     = float;
using ElementSF          = cutlass::float_ue4m3_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

constexpr int AlignmentA = 16 * 8 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 16 * 8 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using TileShape    = Shape<_128, _128, _256>;
using ClusterShape = Shape<_1, _1, _1>;
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

// ── Shared infrastructure ─────────────────────────────────────────

struct ShapeKey {
  int M, N, K;
  bool operator==(const ShapeKey& o) const {
    return M == o.M && N == o.N && K == o.K;
  }
};
struct ShapeKeyHash {
  size_t operator()(const ShapeKey& k) const noexcept {
    return (static_cast<size_t>(k.M) * 1315423911u)
         ^ (static_cast<size_t>(k.N) * 2654435761u)
         ^ static_cast<size_t>(k.K);
  }
};
struct CachedWorkspace { void* ptr = nullptr; size_t size = 0; };

// ── Cooperative schedule ──────────────────────────────────────────

namespace coop {

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
        ElementPairA, LayoutA, AlignmentA,
        ElementPairB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

std::unordered_map<ShapeKey, CachedWorkspace, ShapeKeyHash> g_ws;
std::mutex g_mu;

void* get_ws(int M, int N, int K, size_t need) {
  std::lock_guard<std::mutex> lk(g_mu);
  ShapeKey key{M, N, K};
  auto it = g_ws.find(key);
  if (it != g_ws.end() && it->second.size >= need) return it->second.ptr;
  if (it != g_ws.end()) { cudaFree(it->second.ptr); g_ws.erase(it); }
  CachedWorkspace w; w.size = need;
  if (need > 0) cudaMalloc(&w.ptr, need);
  g_ws[key] = w;
  return w.ptr;
}

cutlass::Status run(
    const void* A_packed, const void* B_packed,
    void* D_bf16, const void* residual_bf16,
    int M, int N, int K,
    const void* SFA, const void* SFB,
    float alpha, cudaStream_t stream)
{
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA strA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB strB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideC strC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD strD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto problem = cute::make_shape(M, N, K, 1);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem);
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem);

  using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
  using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;

  float beta = (residual_bf16 != nullptr) ? 1.0f : 0.0f;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          reinterpret_cast<ArrayElementA const*>(A_packed), strA,
          reinterpret_cast<ArrayElementB const*>(B_packed), strB,
          reinterpret_cast<ElementSF const*>(SFA), layout_SFA,
          reinterpret_cast<ElementSF const*>(SFB), layout_SFB
      },
      {
          {alpha, beta},
          reinterpret_cast<ElementC const*>(residual_bf16), strC,
          reinterpret_cast<ElementD*>(D_bf16), strD
      }
  };

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(args);
  void* ws_ptr = get_ws(M, N, K, ws_size);

  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_gemm_residual_bf16out] can_implement FAIL M=%d N=%d K=%d status=%d\n",
        M, N, K, static_cast<int>(status));
    return status;
  }
  status = gemm.initialize(args, ws_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_gemm_residual_bf16out] initialize FAIL M=%d N=%d K=%d status=%d\n",
        M, N, K, static_cast<int>(status));
    return status;
  }
  return gemm.run(stream);
}

}  // namespace coop

// ── Pingpong schedule ─────────────────────────────────────────────

namespace pingpong {

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
        ElementPairA, LayoutA, AlignmentA,
        ElementPairB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

std::unordered_map<ShapeKey, CachedWorkspace, ShapeKeyHash> g_ws;
std::mutex g_mu;

void* get_ws(int M, int N, int K, size_t need) {
  std::lock_guard<std::mutex> lk(g_mu);
  ShapeKey key{M, N, K};
  auto it = g_ws.find(key);
  if (it != g_ws.end() && it->second.size >= need) return it->second.ptr;
  if (it != g_ws.end()) { cudaFree(it->second.ptr); g_ws.erase(it); }
  CachedWorkspace w; w.size = need;
  if (need > 0) cudaMalloc(&w.ptr, need);
  g_ws[key] = w;
  return w.ptr;
}

cutlass::Status run(
    const void* A_packed, const void* B_packed,
    void* D_bf16, const void* residual_bf16,
    int M, int N, int K,
    const void* SFA, const void* SFB,
    float alpha, cudaStream_t stream)
{
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA strA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB strB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideC strC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD strD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto problem = cute::make_shape(M, N, K, 1);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem);
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem);

  using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
  using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;

  float beta = (residual_bf16 != nullptr) ? 1.0f : 0.0f;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          reinterpret_cast<ArrayElementA const*>(A_packed), strA,
          reinterpret_cast<ArrayElementB const*>(B_packed), strB,
          reinterpret_cast<ElementSF const*>(SFA), layout_SFA,
          reinterpret_cast<ElementSF const*>(SFB), layout_SFB
      },
      {
          {alpha, beta},
          reinterpret_cast<ElementC const*>(residual_bf16), strC,
          reinterpret_cast<ElementD*>(D_bf16), strD
      }
  };

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(args);
  void* ws_ptr = get_ws(M, N, K, ws_size);

  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_gemm_residual_bf16out_pingpong] can_implement FAIL M=%d N=%d K=%d status=%d\n",
        M, N, K, static_cast<int>(status));
    return status;
  }
  status = gemm.initialize(args, ws_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_gemm_residual_bf16out_pingpong] initialize FAIL M=%d N=%d K=%d status=%d\n",
        M, N, K, static_cast<int>(status));
    return status;
  }
  return gemm.run(stream);
}

}  // namespace pingpong

}  // namespace

// ── Public API ────────────────────────────────────────────────────

void fp4_w4a16_gemm_residual_bf16out_sm120(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    const void*  residual_bf16,
    int M, int N, int K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream)
{
  auto status = coop::run(
      A_packed, B_packed, D_bf16, residual_bf16,
      M, N, K, SFA, SFB, alpha, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_gemm_residual_bf16out] run FAIL M=%d N=%d K=%d\n", M, N, K);
  }
}

void fp4_w4a16_gemm_residual_bf16out_sm120_pingpong(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    const void*  residual_bf16,
    int M, int N, int K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream)
{
  auto status = pingpong::run(
      A_packed, B_packed, D_bf16, residual_bf16,
      M, N, K, SFA, SFB, alpha, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_gemm_residual_bf16out_pingpong] run FAIL M=%d N=%d K=%d\n", M, N, K);
  }
}

}  // namespace gemm
}  // namespace flash_rt
