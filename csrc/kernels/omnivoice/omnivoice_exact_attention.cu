#include "omnivoice_exact_attention.cuh"

#include <tuple>

#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>

#include <mutex>
#include <stdexcept>

namespace flash_rt::kernels {
namespace {

using ExactAttentionKernel = PyTorchMemEffAttention::AttentionKernel<
    cutlass::bfloat16_t,
    cutlass::arch::Sm80,
    true,
    32,
    128,
    128,
    // Dropout support is compile-time dead in inference. Bias support stays
    // enabled because its all-masked-row branch is part of PyTorch's exact
    // generated kernel and removing it perturbs BF16 results on some inputs.
    false,
    true>;

__global__ void __launch_bounds__(
    ExactAttentionKernel::kNumThreads,
    ExactAttentionKernel::kMinBlocksPerSm)
omnivoice_exact_packed_attention_kernel(
    ExactAttentionKernel::Params params) {
    if (!params.advance_to_block()) {
        return;
    }
    ExactAttentionKernel::attention_kernel(params);
}

__global__ void __launch_bounds__(
    ExactAttentionKernel::kNumThreads,
    ExactAttentionKernel::kMinBlocksPerSm)
omnivoice_exact_packed_attention_compact_kernel(
    ExactAttentionKernel::Params params,
    int first_blocks,
    int first_length,
    int second_length) {
    const int compact_block = blockIdx.x;
    const int sequence = compact_block >= first_blocks;
    const int query_block = (
        sequence ? compact_block - first_blocks : compact_block);
    const int sequence_start = sequence ? first_length : 0;
    const int sequence_length = sequence ? second_length : first_length;
    const int query_start =
        query_block * ExactAttentionKernel::kQueriesPerBlock;
    const int head = blockIdx.y;

    params.query_ptr +=
        int64_t(sequence_start + query_start) * params.q_strideM +
        head * params.q_strideH;
    params.key_ptr +=
        int64_t(sequence_start) * params.k_strideM +
        head * params.k_strideH;
    params.value_ptr +=
        int64_t(sequence_start) * params.v_strideM +
        head * params.v_strideH;
    params.output_ptr +=
        int64_t(sequence_start + query_start) * params.o_strideM +
        head * params.head_dim_value;
    params.output_accum_ptr =
        reinterpret_cast<ExactAttentionKernel::accum_t*>(
            params.output_ptr);
    params.num_queries = sequence_length - query_start;
    params.num_keys = sequence_length;
    params.num_keys_absolute = sequence_length;
    params.num_batches = 0;
    ExactAttentionKernel::attention_kernel(params);
}

ExactAttentionKernel::Params make_params(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    __nv_bfloat16* output,
    const int32_t* cu_seqlens,
    int total_rows,
    int max_seqlen,
    int num_sequences,
    int num_heads,
    int head_dim,
    float scale) {
    ExactAttentionKernel::Params params;
    params.query_ptr =
        reinterpret_cast<const cutlass::bfloat16_t*>(query);
    params.key_ptr =
        reinterpret_cast<const cutlass::bfloat16_t*>(key);
    params.value_ptr =
        reinterpret_cast<const cutlass::bfloat16_t*>(value);
    params.output_ptr =
        reinterpret_cast<cutlass::bfloat16_t*>(output);
    params.output_accum_ptr = nullptr;
    params.logsumexp_ptr = nullptr;
    params.seqstart_q_ptr = cu_seqlens;
    params.seqstart_k_ptr = cu_seqlens;
    params.num_heads = num_heads;
    params.head_dim = head_dim;
    params.head_dim_value = head_dim;
    params.num_queries = max_seqlen;
    params.num_keys = 0;
    params.num_batches = num_sequences;
    params.custom_mask_type = ExactAttentionKernel::NoCustomMask;
    params.scale = scale;

    const int row_stride = num_heads * head_dim;
    params.q_strideB = static_cast<int64_t>(total_rows) * row_stride;
    params.k_strideB = params.q_strideB;
    params.v_strideB = params.q_strideB;
    params.q_strideM = row_stride;
    params.k_strideM = row_stride;
    params.v_strideM = row_stride;
    params.q_strideH = head_dim;
    params.k_strideH = head_dim;
    params.v_strideH = head_dim;
    params.o_strideM = row_stride;
    return params;
}

template <auto Kernel>
void configure_shared_memory() {
    constexpr size_t shared_bytes =
        sizeof(ExactAttentionKernel::SharedStorage);
    if constexpr (shared_bytes > 0xc000) {
        static std::once_flag shared_memory_flag;
        std::call_once(shared_memory_flag, [] {
            const cudaError_t status = cudaFuncSetAttribute(
                Kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_bytes);
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    "Unable to configure OmniVoice exact attention "
                    "shared memory");
            }
        });
    }
}

}  // namespace

void omnivoice_exact_packed_attention_bf16(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    __nv_bfloat16* output,
    const int32_t* cu_seqlens,
    int total_rows,
    int max_seqlen,
    int num_sequences,
    int num_heads,
    int head_dim,
    float scale,
    cudaStream_t stream) {
    if (query == nullptr || key == nullptr || value == nullptr ||
        output == nullptr || cu_seqlens == nullptr ||
        total_rows <= 0 || max_seqlen <= 0 || num_sequences <= 0 ||
        num_heads <= 0 || head_dim != 128) {
        throw std::invalid_argument(
            "OmniVoice exact packed attention received invalid arguments");
    }

    ExactAttentionKernel::Params params = make_params(
        query, key, value, output, cu_seqlens, total_rows, max_seqlen,
        num_sequences, num_heads, head_dim, scale);
    constexpr size_t shared_bytes =
        sizeof(ExactAttentionKernel::SharedStorage);
    configure_shared_memory<omnivoice_exact_packed_attention_kernel>();
    const dim3 blocks(
        (max_seqlen + ExactAttentionKernel::kQueriesPerBlock - 1) /
            ExactAttentionKernel::kQueriesPerBlock,
        num_heads,
        num_sequences);
    omnivoice_exact_packed_attention_kernel<<<
        blocks,
        params.getThreadsGrid(),
        shared_bytes,
        stream>>>(params);
}

void omnivoice_exact_packed_attention_compact_bf16(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    __nv_bfloat16* output,
    int total_rows,
    int max_seqlen,
    int num_heads,
    int head_dim,
    float scale,
    cudaStream_t stream) {
    const int second_length = total_rows - max_seqlen;
    if (query == nullptr || key == nullptr || value == nullptr ||
        output == nullptr || max_seqlen <= 0 || second_length <= 0 ||
        num_heads <= 0 || head_dim != 128) {
        throw std::invalid_argument(
            "OmniVoice compact attention received invalid arguments");
    }
    ExactAttentionKernel::Params params = make_params(
        query, key, value, output, nullptr, total_rows, max_seqlen,
        2, num_heads, head_dim, scale);
    const int first_blocks =
        (max_seqlen + ExactAttentionKernel::kQueriesPerBlock - 1) /
        ExactAttentionKernel::kQueriesPerBlock;
    const int second_blocks =
        (second_length + ExactAttentionKernel::kQueriesPerBlock - 1) /
        ExactAttentionKernel::kQueriesPerBlock;
    constexpr size_t shared_bytes =
        sizeof(ExactAttentionKernel::SharedStorage);
    configure_shared_memory<
        omnivoice_exact_packed_attention_compact_kernel>();
    const dim3 blocks(first_blocks + second_blocks, num_heads, 1);
    omnivoice_exact_packed_attention_compact_kernel<<<
        blocks,
        params.getThreadsGrid(),
        shared_bytes,
        stream>>>(
            params, first_blocks, max_seqlen, second_length);
}

}  // namespace flash_rt::kernels
