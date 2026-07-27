#include "omnivoice_bf16_dual_swiglu.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/numeric_conversion.h"
#include "device/dual_gemm.h"

namespace flash_rt::kernels {
namespace {

using Element = cutlass::bfloat16_t;
using Accumulator = float;
using Compute = float;
constexpr int kElementsPerAccess = 8;

using ProjectionOutput = cutlass::epilogue::thread::LinearCombination<
    Element,
    kElementsPerAccess,
    Accumulator,
    Compute,
    cutlass::epilogue::thread::ScaleType::Nothing>;

template <typename ElementOutput, int Count>
class ExactRoundedSiluMul {
public:
    using ElementAccumulator = ElementOutput;
    using ElementCompute = float;
    using FragmentOutput = cutlass::Array<ElementOutput,Count>;
    using FragmentAccumulator =
        cutlass::Array<ElementAccumulator,Count>;
    static int const kCount=Count;
    struct Params {};

    CUTLASS_HOST_DEVICE
    ExactRoundedSiluMul(Params const& = Params{}) {}

    CUTLASS_HOST_DEVICE
    bool is_source_needed() const { return false; }

    CUTLASS_HOST_DEVICE
    void set_k_partition(int, int) {}

    CUTLASS_DEVICE
    FragmentOutput operator()(
        FragmentAccumulator const& gate,
        FragmentAccumulator const& up) const {
        FragmentOutput result;
        cutlass::NumericConverter<float,ElementOutput> to_float;
        cutlass::NumericConverter<ElementOutput,float> to_bf16;
        CUTLASS_PRAGMA_UNROLL
        for (int i=0;i<Count;++i) {
            float gate_f=to_float(gate[i]);
            float up_f=to_float(up[i]);
            float silu=gate_f/(1.0f+expf(-gate_f));
            float silu_rounded=to_float(to_bf16(silu));
            result[i]=to_bf16(silu_rounded*up_f);
        }
        return result;
    }
};

using SwiGLUOutput = ExactRoundedSiluMul<Element,kElementsPerAccess>;
using InstructionShape = cutlass::gemm::GemmShape<16,8,16>;

template <typename ThreadblockShape,typename WarpShape,int Stages>
using DualSwiGLU = cutlass::gemm::device::DualGemm<
    Element,
    cutlass::layout::RowMajor,
    Element,
    cutlass::layout::ColumnMajor,
    cutlass::layout::ColumnMajor,
    Element,
    cutlass::layout::RowMajor,
    Accumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,WarpShape,
    InstructionShape,
    ProjectionOutput,
    ProjectionOutput,
    SwiGLUOutput,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    Stages,
    false,
    false,
    false,
    8,
    8>;

using Dual64x64 = DualSwiGLU<
    cutlass::gemm::GemmShape<64,64,32>,
    cutlass::gemm::GemmShape<32,32,32>,3>;

template <typename Op>
int run_dual_swiglu(
    const __nv_bfloat16* x,
    const __nv_bfloat16* gate_weight,
    const __nv_bfloat16* up_weight,
    __nv_bfloat16* out,
    int rows,
    int hidden_dim,
    int input_dim,
    cudaStream_t stream) {
    using TensorRefC =
        cutlass::TensorRef<Element,cutlass::layout::RowMajor>;
    const cutlass::gemm::GemmCoord problem(
        rows,hidden_dim,input_dim);
    const typename Op::TensorRefA ref_a(
        reinterpret_cast<const Element*>(x),
        cutlass::layout::RowMajor(input_dim));
    const typename Op::TensorRefB0 ref_gate(
        reinterpret_cast<const Element*>(gate_weight),
        cutlass::layout::ColumnMajor(input_dim));
    const typename Op::TensorRefB1 ref_up(
        reinterpret_cast<const Element*>(up_weight),
        cutlass::layout::ColumnMajor(input_dim));
    const TensorRefC null_ref{};
    const typename Op::TensorRefD ref_out(
        reinterpret_cast<Element*>(out),
        cutlass::layout::RowMajor(hidden_dim));

    typename Op::Arguments args{
        cutlass::gemm::DualGemmMode::kGemm,
        problem,
        ref_a,
        ref_gate,
        null_ref,
        null_ref,
        ref_up,
        null_ref,
        null_ref,
        ref_out,
        {1.0f,0.0f},
        {1.0f,0.0f},
        {},
        1,
        1,
        0,0,0,0,0};

    Op op;
    if (op.get_workspace_size(args)!=0) {
        return -2;
    }
    cutlass::Status status=op.can_implement(args);
    if (status!=cutlass::Status::kSuccess) {
        return -10-static_cast<int>(status);
    }
    status=op.initialize(args,nullptr,stream);
    if (status!=cutlass::Status::kSuccess) {
        return -20-static_cast<int>(status);
    }
    status=op.run(stream);
    return status==cutlass::Status::kSuccess
        ? 0 : -30-static_cast<int>(status);
}

}  // namespace

int omnivoice_bf16_dual_swiglu(
    const __nv_bfloat16* x,
    const __nv_bfloat16* gate_weight,
    const __nv_bfloat16* up_weight,
    __nv_bfloat16* out,
    int rows,
    int hidden_dim,
    int input_dim,
    cudaStream_t stream) {
    if (x==nullptr || gate_weight==nullptr || up_weight==nullptr ||
        out==nullptr || rows<=0 || hidden_dim<=0 || input_dim<=0) {
        return -1;
    }

    return run_dual_swiglu<Dual64x64>(
        x,gate_weight,up_weight,out,rows,hidden_dim,input_dim,stream);
}

}  // namespace flash_rt::kernels
