// ============================================================================
//  FlashRT — NVFP4 GEMM with fused silu(C_src) * acc → fp4 + SFA.
//
//  Uses Sm90SrcFetch (C source matrix = gate values) to avoid the
//  Sm90AuxLoad null-pointer bug in CUTLASS 4.3.1.
//
//  EVT tree:
//    Sm90EVT<
//        Sm100BlockScaleFactorRowStore<SFVecSize=16, ...>,
//        Sm90EVT<
//            Sm90Compute<multiplies, ...>,          // silu(C) * acc
//            Sm90EVT<
//                Sm90Compute<SiLu, ...>,             // silu(C)
//                Sm90SrcFetch<ElementSource>         // C = gate_bf16
//            >,
//            Sm90AccFetch                            // acc
//        >
//    >
//
//  STATUS: NEW — using Sm90SrcFetch instead of Sm90AuxLoad.
// ============================================================================
#include "gemm/fp4/cutlass_fp4_gemm_silu_src.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Custom FusionOp: D = blockscale(silu(C) * acc)
//   C (= gate_bf16) is the source matrix (IsSourceSupported=true)
//   NO aux load (IsAuxInSupported=false)
//   Block scale output (IsBlockScaleSupported=true)
//
/////////////////////////////////////////////////////////////////////////////////////////////////
template<
  int SFVecSize_,
  class ElementOutput_,                  // e.g. cutlass::float_e2m1_t
  class ElementCompute_,                 // e.g. float
  class ElementBlockScaleFactor_,        // e.g. cutlass::float_ue4m3_t
  class ElementSource_ = ElementOutput_, // gate: e.g. cutlass::bfloat16_t
  class ElementScalar_ = ElementCompute_,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct SiLuMulSrcBlockScaleFactor : FusionOperation {
  using ElementOutput            = ElementOutput_;
  using ElementCompute           = ElementCompute_;
  using ElementSource            = ElementSource_;
  using ElementScalar            = ElementScalar_;
  using ElementBlockScaleFactor  = ElementBlockScaleFactor_;

  static constexpr int SFVecSize     = SFVecSize_;
  static constexpr FloatRoundStyle RoundStyle = RoundStyle_;
  static constexpr bool IsSourceSupported     = true;   // use Sm90SrcFetch
  static constexpr bool IsAuxOutSupported     = false;
  static constexpr bool IsAuxInSupported      = false;  // NO Sm90AuxLoad!
  static constexpr bool IsBlockScaleSupported = true;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// EVT alias for the Sm100 RowMajor SF case.
//
// EVT tree:
//   Sm90EVT<Sm100BlockScaleFactorRowStore<...>,        // root: pack+store
//     Sm90EVT<Sm90Compute<multiplies, ...>,             // multiply
//       Sm90EVT<Sm90Compute<SiLu, ...>,                 // silu
//         Sm90SrcFetch<ElementSource>                   // C = gate
//       >,
//       Sm90AccFetch                                     // acc
//     >
//   >
//
/////////////////////////////////////////////////////////////////////////////////////////////////
template<
  int SFVecSize,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  class ElementSource,
  FloatRoundStyle RoundStyle
>
using Sm100SiLuMulSrcRowBlockScaleFactor =
  Sm90EVT<
    Sm100BlockScaleFactorRowStore<
        SFVecSize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor, RoundStyle>,
    Sm90EVT<
      Sm90Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>,
      Sm90EVT<
        Sm90Compute<epilogue::thread::SiLu, ElementCompute, ElementCompute, RoundStyle>,
        Sm90SrcFetch<ElementSource>
      >,
      Sm90AccFetch
    >
  >;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// FusionCallbacks specialization for Sm100TmaWarpSpecialized.
// Maps SiLuMulSrcBlockScaleFactor to the EVT impl above.
//
// Note: Args... captures SmemLayoutAtomC and CopyOpS2RC injected by CallbacksBuilder.
// These are required by Sm90SrcFetch for C-source TMA loading.
//
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int  SFVecSize,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class... Args
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::SiLuMulSrcBlockScaleFactor<
        SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor,
        ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile,
    Args...
> : Sm100SiLuMulSrcRowBlockScaleFactor<
        SFVecSize, EpilogueTile,
        typename detail::get_unpacked_element_type<ElementOutput>::type,
        ElementCompute, ElementBlockScaleFactor,
        ElementSource, RoundStyle>
{
  using Impl = Sm100SiLuMulSrcRowBlockScaleFactor<
      SFVecSize, EpilogueTile,
      typename detail::get_unpacked_element_type<ElementOutput>::type,
      ElementCompute, ElementBlockScaleFactor,
      ElementSource, RoundStyle>;
  using Operation = fusion::SiLuMulSrcBlockScaleFactor<
      SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor,
      ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    // Block-scaled FP4 output config
    ElementBlockScaleFactor* block_scale_factor_ptr = nullptr;
    using StrideNormConst = Stride<_0, _0, int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        // Sm90EVT root: BlockScaleFactorRowStore over child subtree
        {
          // Child subtree: silu(C) * acc
          {
            // multiply binary op args: empty (stateless multiplies)
            {},
            // First operand: silu(C)
            {
              // SiLu compute op args: empty (stateless)
              {},
              // Sm90SrcFetch args: empty (C is passed via GEMM epilogue C args)
              {}
            },
            // Second operand: Sm90AccFetch args: empty
            {}
          },
          // BlockScaleFactor store args
          { block_scale_factor_ptr, norm_constant_ptr, dNormConst }
        };
    }
  };

  // Ctor inheritance from Impl
  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// GEMM type definition & entry point
//
/////////////////////////////////////////////////////////////////////////////////////////////////
namespace flash_rt {
namespace fp4 {
namespace silu_src {

using namespace cute;

using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD   = cutlass::float_e2m1_t;            // FP4 packed output
using ElementC   = cutlass::bfloat16_t;              // C source = gate BF16 (≠ D, ok for Sm90SrcFetch)
using LayoutDTag = cutlass::layout::RowMajor;
using LayoutCTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 32;                       // fp4: 128/4 = 32
constexpr int AlignmentC = 8;                        // bf16: 128/16 = 8

using ElementSFD = cutlass::float_ue4m3_t;
using LayoutSFDTag = LayoutDTag;

// Gate values passed as ElementSource (C matrix) — BF16
using ElementSource    = ElementC;                   // = bfloat16_t
using LayoutSourceTag  = LayoutCTag;
constexpr int AlignmentSource = AlignmentC;

using ElementAccumulator  = float;
using ElementCompute      = float;
using ElementScalar       = float;
using ArchTag             = cutlass::arch::Sm100;
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;

constexpr int InputSFVectorSize  = 16;
constexpr int OutputSFVectorSize = InputSFVectorSize;

// Tile / cluster — V8 shape (well-tested for OmniVoice shapes)
using MmaTileShape = Shape<_128, _256, _256>;
using ClusterShape = Shape<_1, _1, _1>;

using FusionOperation = cutlass::epilogue::fusion::SiLuMulSrcBlockScaleFactor<
    OutputSFVectorSize,
    ElementD,
    ElementCompute,
    ElementSFD,
    ElementSource,
    ElementScalar,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperation
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop, CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;

using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

}  // namespace silu_src

int cutlass_fp4_gemm_silu_src_fp4(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void const* C_gate_bf16,
    void*       D_packed,
    void*       D_SFA,
    int M, int N, int K,
    cudaStream_t stream) {
  using namespace silu_src;

  auto stride_A   = cutlass::make_cute_packed_stride(StrideA{},   {M, K, 1});
  auto stride_B   = cutlass::make_cute_packed_stride(StrideB{},   {N, K, 1});
  auto stride_C   = cutlass::make_cute_packed_stride(StrideC{},   {M, N, 1});
  auto stride_D   = cutlass::make_cute_packed_stride(StrideD{},   {M, N, 1});
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

  using EA = typename ElementA::DataType;
  using SA = typename ElementA::ScaleFactorType;
  using EB = typename ElementB::DataType;
  using SB = typename ElementB::ScaleFactorType;

  // Pass gate_bf16 as C source matrix. The epilogue reads it via Sm90SrcFetch.
  // Note: The CUTLASS CollectiveEpilogue expects C and D to have compatible types.
  // Since D is float_e2m1_t (packed) and C is bfloat16_t, the epilogue handles
  // the type conversion internally via Sm90SrcFetch's recast.
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      { /*Mainloop*/
          reinterpret_cast<EA const*>(A_packed), stride_A,
          reinterpret_cast<EB const*>(B_packed), stride_B,
          reinterpret_cast<SA const*>(SFA), layout_SFA,
          reinterpret_cast<SB const*>(SFB), layout_SFB
      },
      { /*Epilogue*/
        { /*FusionCallbacks::Arguments*/
          reinterpret_cast<ElementSFD*>(D_SFA),       // block_scale_factor_ptr
          /*norm_constant_ptr*/ nullptr,
          /*dNormConst*/ {}
        },
        // C source = gate_bf16. The epilogue reads this via Sm90SrcFetch<ElementSource=bf16>.
        const_cast<ElementC*>(reinterpret_cast<ElementC const*>(C_gate_bf16)), stride_C,
        reinterpret_cast<ElementD*>(D_packed), stride_D
      }
  };

  Gemm gemm;
  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) {
    return static_cast<int>(st) | 0x10000;
  }
  size_t ws_sz = Gemm::get_workspace_size(args);
  void* ws = nullptr;
  if (ws_sz > 0 && cudaMalloc(&ws, ws_sz) != cudaSuccess) return -1;
  st = gemm.initialize(args, ws, stream);
  if (st != cutlass::Status::kSuccess) {
    if (ws) cudaFree(ws);
    return static_cast<int>(st) | 0x20000;
  }
  st = gemm.run(stream);
  if (ws) cudaFree(ws);
  return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

}  // namespace fp4
}  // namespace flash_rt
