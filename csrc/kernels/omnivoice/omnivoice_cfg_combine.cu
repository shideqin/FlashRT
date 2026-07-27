// ================================================================
// FlashRT — Fused CFG combine kernel (BF16). See omnivoice_cfg_combine.cuh.
// ================================================================

#include "omnivoice_cfg_combine.cuh"

#include <cub/block/block_radix_sort.cuh>

#define CC_WARP 32
#define CC_ITERS 48   // 48*32 = 1536 cols max (OmniVoice vocab = 1025)

__global__ void omnivoice_cfg_logsoftmax_bf16_kernel(
    const __nv_bfloat16* __restrict__ c_logits,
    const __nv_bfloat16* __restrict__ u_logits,
    __nv_bfloat16* __restrict__ out,
    int cols, int mask_col, float gs)
{
    const int row = blockIdx.x;
    const int lane = threadIdx.x;

    const __nv_bfloat16* cr = c_logits + row * cols;
    const __nv_bfloat16* ur = u_logits + row * cols;
    __nv_bfloat16* outr = out + row * cols;

    float creg[CC_ITERS], ureg[CC_ITERS];

    // ── Pass 1: load c (mask_col → -inf) + u, find max_c, max_u ──
    float mx_c = -1e30f, mx_u = -1e30f;
    #pragma unroll
    for (int it = 0; it < CC_ITERS; ++it) {
        int col = it * CC_WARP + lane;
        float cv = (col < cols) ? __bfloat162float(cr[col]) : -1e30f;
        float uv = (col < cols) ? __bfloat162float(ur[col]) : -1e30f;
        if (col == mask_col) cv = -1e30f;   // mask only c (matches PyTorch ref)
        creg[it] = cv; ureg[it] = uv;
        mx_c = fmaxf(mx_c, cv);
        mx_u = fmaxf(mx_u, uv);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        mx_c = fmaxf(mx_c, __shfl_xor_sync(0xffffffff, mx_c, o));
        mx_u = fmaxf(mx_u, __shfl_xor_sync(0xffffffff, mx_u, o));
    }

    // ── Pass 2: sum exp → lc, lu (log normalizers) ──
    float sc = 0.f, su = 0.f;
    #pragma unroll
    for (int it = 0; it < CC_ITERS; ++it) {
        sc += __expf(creg[it] - mx_c);
        su += __expf(ureg[it] - mx_u);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sc += __shfl_xor_sync(0xffffffff, sc, o);
        su += __shfl_xor_sync(0xffffffff, su, o);
    }
    const float lc = logf(sc + 1e-12f);
    const float lu = logf(su + 1e-12f);

    // ── Pass 3: combined = c_lp + gs*(c_lp - u_lp), find max_comb ──
    // c_lp = creg - mx_c - lc ; u_lp = ureg - mx_u - lu
    float mx_comb = -1e30f;
    #pragma unroll
    for (int it = 0; it < CC_ITERS; ++it) {
        float clp = creg[it] - mx_c - lc;
        float ulp = ureg[it] - mx_u - lu;
        float comb = clp + gs * (clp - ulp);
        creg[it] = comb;             // reuse creg[] to store combined (saves a 3rd array)
        mx_comb = fmaxf(mx_comb, comb);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        mx_comb = fmaxf(mx_comb, __shfl_xor_sync(0xffffffff, mx_comb, o));

    // ── Pass 4: out = log_softmax(combined) ──
    float s_comb = 0.f;
    #pragma unroll
    for (int it = 0; it < CC_ITERS; ++it)
        s_comb += __expf(creg[it] - mx_comb);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        s_comb += __shfl_xor_sync(0xffffffff, s_comb, o);
    const float lcomb = logf(s_comb + 1e-12f);

    #pragma unroll
    for (int it = 0; it < CC_ITERS; ++it) {
        int col = it * CC_WARP + lane;
        if (col < cols)
            outr[col] = __float2bfloat16(creg[it] - mx_comb - lcomb);
    }
}

void omnivoice_cfg_logsoftmax_bf16(
    const __nv_bfloat16* c_logits,
    const __nv_bfloat16* u_logits,
    __nv_bfloat16* out,
    int rows, int cols, int mask_col, float guidance_scale,
    cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;
    dim3 grid(rows), block(CC_WARP);
    omnivoice_cfg_logsoftmax_bf16_kernel<<<grid, block, 0, stream>>>(
        c_logits, u_logits, out, cols, mask_col, guidance_scale);
}

namespace {

__global__ void omnivoice_cfg_combine_f32_kernel(
    const float* __restrict__ conditional_log_probs,
    const float* __restrict__ unconditional_log_probs,
    float* __restrict__ out,
    int elements,
    float guidance_scale)
{
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=elements) return;
    // Keep the same three FP32 rounding points as eager PyTorch:
    // cl + guidance_scale * (cl - ul).
    const float difference=__fsub_rn(
        conditional_log_probs[index],unconditional_log_probs[index]);
    const float scaled=__fmul_rn(guidance_scale,difference);
    out[index]=__fadd_rn(conditional_log_probs[index],scaled);
}

// OmniVoice has 1025 audio-vocabulary columns.  PyTorch dispatches its
// final FP32 log_softmax to PersistentSoftmax's log2_elements=11 kernel:
// one 32-lane warp per row, 64 values per lane, and XOR reductions.  Mirror
// that operation order here while keeping the CFG result in registers, then
// perform the following masked argmax/score reduction in the same launch.
constexpr int kCfgSelectWarp = 32;
constexpr int kCfgSelectIterations = 64;
constexpr int kCfgSelectWarpsPerBlock = 4;
constexpr int kCfgSelectCapacity =
    kCfgSelectWarp * kCfgSelectIterations;

__device__ __forceinline__ float cfg_warp_max(float value)
{
    #pragma unroll
    for (int offset=kCfgSelectWarp/2;offset>0;offset>>=1) {
        const float other=__shfl_xor_sync(
            0xffffffffu,value,offset,kCfgSelectWarp);
        // Match ATen PersistentSoftmax's Max functor exactly.
        value=value<other ? other : value;
    }
    return value;
}

__device__ __forceinline__ float cfg_warp_sum(float value)
{
    #pragma unroll
    for (int offset=kCfgSelectWarp/2;offset>0;offset>>=1) {
        value=__fadd_rn(
            value,__shfl_xor_sync(
                0xffffffffu,value,offset,kCfgSelectWarp));
    }
    return value;
}

__global__ void omnivoice_logits_bf16_cfg_argmax_scores_f32_kernel(
    const __nv_bfloat16* __restrict__ packed_logits,
    const float* __restrict__ layer_penalty,
    int64_t* __restrict__ predictions,
    float* __restrict__ scores,
    int rows,
    int cols,
    int target_length,
    int mask_col,
    float guidance_scale)
{
    const int lane=threadIdx.x;
    const int row=blockIdx.x*blockDim.y+threadIdx.y;
    if (row>=rows) return;

    // packed_logits is [2, target_length, codebooks, cols], while selection
    // is [codebooks, target_length].  Map the logical row without asking
    // PyTorch to materialize the permuted logits.
    const int num_codebooks=rows/target_length;
    const int codebook=row/target_length;
    const int target=row-codebook*target_length;
    const int packed_row=target*num_codebooks+codebook;
    const __nv_bfloat16* conditional=
        packed_logits+static_cast<int64_t>(packed_row)*cols;
    const __nv_bfloat16* unconditional=
        conditional+static_cast<int64_t>(rows)*cols;
    float combined_values[kCfgSelectIterations];
    const float negative_infinity=-__int_as_float(0x7f800000);

    float conditional_max=negative_infinity;
    float unconditional_max=negative_infinity;
    #pragma unroll
    for (int iteration=0;iteration<kCfgSelectIterations;++iteration) {
        const int column=lane+iteration*kCfgSelectWarp;
        const float conditional_value=(
            column<cols
            ? __bfloat162float(conditional[column])
            : negative_infinity);
        const float unconditional_value=(
            column<cols
            ? __bfloat162float(unconditional[column])
            : negative_infinity);
        conditional_max=(
            conditional_max>conditional_value
            ? conditional_max : conditional_value);
        unconditional_max=(
            unconditional_max>unconditional_value
            ? unconditional_max : unconditional_value);
    }
    conditional_max=cfg_warp_max(conditional_max);
    unconditional_max=cfg_warp_max(unconditional_max);

    float conditional_sum=0.0f;
    float unconditional_sum=0.0f;
    #pragma unroll
    for (int iteration=0;iteration<kCfgSelectIterations;++iteration) {
        const int column=lane+iteration*kCfgSelectWarp;
        const float conditional_value=(
            column<cols
            ? __bfloat162float(conditional[column])
            : negative_infinity);
        const float unconditional_value=(
            column<cols
            ? __bfloat162float(unconditional[column])
            : negative_infinity);
        conditional_sum=__fadd_rn(
            conditional_sum,expf(__fsub_rn(
                conditional_value,conditional_max)));
        unconditional_sum=__fadd_rn(
            unconditional_sum,expf(__fsub_rn(
                unconditional_value,unconditional_max)));
    }
    conditional_sum=cfg_warp_sum(conditional_sum);
    unconditional_sum=cfg_warp_sum(unconditional_sum);
    const float conditional_log_sum=logf(conditional_sum);
    const float unconditional_log_sum=logf(unconditional_sum);

    float combined_max=negative_infinity;
    #pragma unroll
    for (int iteration=0;iteration<kCfgSelectIterations;++iteration) {
        const int column=lane+iteration*kCfgSelectWarp;
        float value=negative_infinity;
        if (column<cols) {
            const float conditional_value=
                __bfloat162float(conditional[column]);
            const float unconditional_value=
                __bfloat162float(unconditional[column]);
            const float conditional_log_probability=__fsub_rn(
                __fsub_rn(
                    conditional_value,conditional_max),
                conditional_log_sum);
            const float unconditional_log_probability=__fsub_rn(
                __fsub_rn(
                    unconditional_value,unconditional_max),
                unconditional_log_sum);
            const float difference=__fsub_rn(
                conditional_log_probability,unconditional_log_probability);
            const float scaled=__fmul_rn(guidance_scale,difference);
            value=__fadd_rn(conditional_log_probability,scaled);
        }
        combined_values[iteration]=value;
        combined_max=combined_max>value ? combined_max : value;
    }
    combined_max=cfg_warp_max(combined_max);

    float combined_sum=0.0f;
    #pragma unroll
    for (int iteration=0;iteration<kCfgSelectIterations;++iteration) {
        combined_sum=__fadd_rn(
            combined_sum,expf(__fsub_rn(
                combined_values[iteration],combined_max)));
    }
    combined_sum=cfg_warp_sum(combined_sum);
    const float combined_log_sum=logf(combined_sum);

    float best=negative_infinity;
    int best_index=0;
    #pragma unroll
    for (int iteration=0;iteration<kCfgSelectIterations;++iteration) {
        const int column=lane+iteration*kCfgSelectWarp;
        if (column<cols && column!=mask_col) {
            const float log_probability=__fsub_rn(
                __fsub_rn(
                    combined_values[iteration],combined_max),
                combined_log_sum);
            if (log_probability>best ||
                (log_probability==best && column<best_index)) {
                best=log_probability;
                best_index=column;
            }
        }
    }
    #pragma unroll
    for (int offset=kCfgSelectWarp/2;offset>0;offset>>=1) {
        const float other=__shfl_xor_sync(
            0xffffffffu,best,offset,kCfgSelectWarp);
        const int other_index=__shfl_xor_sync(
            0xffffffffu,best_index,offset,kCfgSelectWarp);
        if (other>best || (other==best && other_index<best_index)) {
            best=other;
            best_index=other_index;
        }
    }
    if (lane==0) {
        predictions[row]=best_index;
        scores[row]=__fsub_rn(
            best,layer_penalty[row/target_length]);
    }
}

__global__ void omnivoice_embed_codebooks_bf16_kernel(
    const int64_t* __restrict__ tokens,
    const __nv_bfloat16* __restrict__ embedding,
    const int64_t* __restrict__ codebook_offsets,
    __nv_bfloat16* __restrict__ conditional_out,
    __nv_bfloat16* __restrict__ unconditional_out,
    int num_codebooks,
    int token_stride,
    int elements,
    int embedding_dim)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    for (;index<elements;index+=blockDim.x*gridDim.x) {
        const int position=index/embedding_dim;
        const int column=index-position*embedding_dim;
        float sum=0.0f;
        #pragma unroll
        for (int codebook=0;codebook<8;++codebook) {
            if (codebook<num_codebooks) {
                const int64_t token=tokens[
                    codebook*token_stride+position];
                const int64_t row=token+codebook_offsets[codebook];
                sum=__fadd_rn(
                    sum,__bfloat162float(
                        embedding[row*embedding_dim+column]));
            }
        }
        const __nv_bfloat16 value=__float2bfloat16_rn(sum);
        conditional_out[index]=value;
        unconditional_out[index]=value;
    }
}

__global__ void omnivoice_prepare_cfg_embeddings_bf16_kernel(
    const int64_t* __restrict__ tokens,
    const __nv_bfloat16* __restrict__ embedding,
    const int64_t* __restrict__ codebook_offsets,
    const __nv_bfloat16* __restrict__ base_input,
    __nv_bfloat16* __restrict__ out,
    int num_codebooks,
    int token_stride,
    int sequence_length,
    int target_length,
    int embedding_dim,
    int elements)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    for (;index<elements;index+=blockDim.x*gridDim.x) {
        const int row=index/embedding_dim;
        const int column=index-row*embedding_dim;
        int token_position=-1;
        if (row>=sequence_length-target_length &&
            row<sequence_length) {
            token_position=row-(sequence_length-target_length);
        } else if (row>=sequence_length &&
                   row<sequence_length+target_length) {
            token_position=row-sequence_length;
        }
        if (token_position<0) {
            out[index]=base_input[index];
            continue;
        }
        float sum=0.0f;
        #pragma unroll
        for (int codebook=0;codebook<8;++codebook) {
            if (codebook<num_codebooks) {
                const int64_t token=tokens[
                    codebook*token_stride+token_position];
                const int64_t embedding_row=
                    token+codebook_offsets[codebook];
                sum=__fadd_rn(
                    sum,__bfloat162float(
                        embedding[
                            embedding_row*embedding_dim+column]));
            }
        }
        out[index]=__float2bfloat16_rn(sum);
    }
}

__global__ void omnivoice_argmax_scores_f32_kernel(
    const float* __restrict__ log_probs,
    const float* __restrict__ layer_penalty,
    int64_t* __restrict__ predictions,
    float* __restrict__ scores,
    int cols,
    int target_length,
    int mask_col)
{
    __shared__ float shared_values[256];
    __shared__ int shared_indices[256];
    const int row=blockIdx.x;
    const float* row_probs=log_probs+row*cols;
    float best=-__int_as_float(0x7f800000);
    int best_index=0;
    for (int col=threadIdx.x;col<cols;col+=blockDim.x) {
        const float value=(
            col==mask_col
            ? -__int_as_float(0x7f800000)
            : row_probs[col]);
        if (value>best || (value==best && col<best_index)) {
            best=value;
            best_index=col;
        }
    }
    shared_values[threadIdx.x]=best;
    shared_indices[threadIdx.x]=best_index;
    __syncthreads();

    #pragma unroll
    for (int offset=128;offset>0;offset>>=1) {
        if (threadIdx.x<offset) {
            const float other=shared_values[threadIdx.x+offset];
            const int other_index=shared_indices[threadIdx.x+offset];
            if (other>shared_values[threadIdx.x] ||
                (other==shared_values[threadIdx.x] &&
                 other_index<shared_indices[threadIdx.x])) {
                shared_values[threadIdx.x]=other;
                shared_indices[threadIdx.x]=other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x==0) {
        predictions[row]=shared_indices[0];
        scores[row]=__fsub_rn(
            shared_values[0],layer_penalty[row/target_length]);
    }
}

constexpr int kSelectThreads = 128;
constexpr int kSelectSmallItems = 4;
constexpr int kSelectLargeItems = 8;
constexpr int kSelectSmallCapacity =
    kSelectThreads * kSelectSmallItems;
constexpr int kSelectLargeCapacity =
    kSelectThreads * kSelectLargeItems;

template <int ItemsPerThread>
__global__ void omnivoice_maskgit_topk_update_kernel(
    const float* __restrict__ scores,
    const int64_t* __restrict__ predictions,
    int64_t* __restrict__ tokens,
    int elements,
    int k,
    const int* __restrict__ schedule,
    int* __restrict__ step,
    int schedule_length,
    int64_t mask_id)
{
    using BlockSort = cub::BlockRadixSort<
        float, kSelectThreads, ItemsPerThread, int>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    float thread_scores[ItemsPerThread];
    int thread_indices[ItemsPerThread];
    const int current_step=schedule == nullptr ? 0 : *step;
    if (schedule != nullptr) {
        k=(current_step>=0 && current_step<schedule_length)
            ? schedule[current_step] : 0;
    }

    #pragma unroll
    for (int item=0;item<ItemsPerThread;++item) {
        int idx=threadIdx.x*ItemsPerThread+item;
        bool valid=idx<elements && tokens[idx]==mask_id;
        thread_scores[item]=
            valid ? scores[idx] : -__int_as_float(0x7f800000);
        thread_indices[item]=idx;
    }
    BlockSort(sort_storage).SortDescending(
        thread_scores,thread_indices);
    #pragma unroll
    for (int item=0;item<ItemsPerThread;++item) {
        int rank=threadIdx.x*ItemsPerThread+item;
        int idx=thread_indices[item];
        if (rank<k && idx<elements) {
            tokens[idx]=predictions[idx];
        }
    }
    if (schedule != nullptr && threadIdx.x==0) {
        *step=current_step+1;
    }
}

template <int ItemsPerThread>
__global__ void omnivoice_maskgit_topk_gumbel_update_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ uniforms,
    const int64_t* __restrict__ predictions,
    int64_t* __restrict__ tokens,
    int elements,
    int k,
    const int* __restrict__ schedule,
    int* __restrict__ step,
    int schedule_length,
    int64_t mask_id,
    float temperature)
{
    using BlockSort = cub::BlockRadixSort<
        float, kSelectThreads, ItemsPerThread, int>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    float thread_scores[ItemsPerThread];
    int thread_indices[ItemsPerThread];
    const int current_step=schedule == nullptr ? 0 : *step;
    if (schedule != nullptr) {
        k=(current_step>=0 && current_step<schedule_length)
            ? schedule[current_step] : 0;
    }

    #pragma unroll
    for (int item=0;item<ItemsPerThread;++item) {
        const int idx=threadIdx.x*ItemsPerThread+item;
        const bool valid=idx<elements && tokens[idx]==mask_id;
        float value=-__int_as_float(0x7f800000);
        if (valid) {
            const float inner_log=logf(
                __fadd_rn(uniforms[idx],1.0e-10f));
            const float outer_arg=__fadd_rn(-inner_log,1.0e-10f);
            const float noise=-logf(outer_arg);
            value=__fadd_rn(scores[idx]/temperature,noise);
        }
        thread_scores[item]=value;
        thread_indices[item]=idx;
    }
    BlockSort(sort_storage).SortDescending(
        thread_scores,thread_indices);
    #pragma unroll
    for (int item=0;item<ItemsPerThread;++item) {
        const int rank=threadIdx.x*ItemsPerThread+item;
        const int idx=thread_indices[item];
        if (rank<k && idx<elements) {
            tokens[idx]=predictions[idx];
        }
    }
    if (schedule != nullptr && threadIdx.x==0) {
        *step=current_step+1;
    }
}

template <int SmallItems, int LargeItems, typename... Args>
void launch_maskgit_topk(
    int elements,
    cudaStream_t stream,
    Args... args) {
    if (elements<=kSelectSmallCapacity) {
        omnivoice_maskgit_topk_update_kernel<SmallItems>
            <<<1,kSelectThreads,0,stream>>>(args...);
    } else {
        omnivoice_maskgit_topk_update_kernel<LargeItems>
            <<<1,kSelectThreads,0,stream>>>(args...);
    }
}

template <int SmallItems, int LargeItems, typename... Args>
void launch_maskgit_topk_gumbel(
    int elements,
    cudaStream_t stream,
    Args... args) {
    if (elements<=kSelectSmallCapacity) {
        omnivoice_maskgit_topk_gumbel_update_kernel<SmallItems>
            <<<1,kSelectThreads,0,stream>>>(args...);
    } else {
        omnivoice_maskgit_topk_gumbel_update_kernel<LargeItems>
            <<<1,kSelectThreads,0,stream>>>(args...);
    }
}

}  // namespace

void omnivoice_cfg_combine_f32(
    const float* conditional_log_probs,
    const float* unconditional_log_probs,
    float* out,
    int elements,
    float guidance_scale,
    cudaStream_t stream)
{
    if (elements<=0) return;
    constexpr int threads=256;
    const int blocks=(elements+threads-1)/threads;
    omnivoice_cfg_combine_f32_kernel<<<blocks,threads,0,stream>>>(
        conditional_log_probs,unconditional_log_probs,out,
        elements,guidance_scale);
}

void omnivoice_logits_bf16_cfg_argmax_scores_f32(
    const __nv_bfloat16* packed_logits,
    const float* layer_penalty,
    int64_t* predictions,
    float* scores,
    int rows,
    int cols,
    int target_length,
    int mask_col,
    float guidance_scale,
    cudaStream_t stream)
{
    if (rows<=0 || cols<=0 || cols>kCfgSelectCapacity ||
        target_length<=0 || rows%target_length!=0) {
        return;
    }
    const dim3 block(kCfgSelectWarp,kCfgSelectWarpsPerBlock);
    const dim3 grid(
        (rows+kCfgSelectWarpsPerBlock-1)/kCfgSelectWarpsPerBlock);
    omnivoice_logits_bf16_cfg_argmax_scores_f32_kernel
        <<<grid,block,0,stream>>>(
            packed_logits,layer_penalty,predictions,scores,
            rows,cols,target_length,mask_col,guidance_scale);
}

void omnivoice_embed_codebooks_bf16(
    const int64_t* tokens,
    const __nv_bfloat16* embedding,
    const int64_t* codebook_offsets,
    __nv_bfloat16* conditional_out,
    __nv_bfloat16* unconditional_out,
    int num_codebooks,
    int token_stride,
    int target_length,
    int embedding_dim,
    cudaStream_t stream)
{
    const int elements=target_length*embedding_dim;
    if (elements<=0 || num_codebooks<=0 || num_codebooks>8) return;
    constexpr int threads=256;
    int blocks=(elements+threads-1)/threads;
    if (blocks>4096) blocks=4096;
    omnivoice_embed_codebooks_bf16_kernel<<<blocks,threads,0,stream>>>(
        tokens,embedding,codebook_offsets,
        conditional_out,unconditional_out,
        num_codebooks,token_stride,elements,embedding_dim);
}

void omnivoice_prepare_cfg_embeddings_bf16(
    const int64_t* tokens,
    const __nv_bfloat16* embedding,
    const int64_t* codebook_offsets,
    const __nv_bfloat16* base_input,
    __nv_bfloat16* out,
    int num_codebooks,
    int token_stride,
    int sequence_length,
    int target_length,
    int embedding_dim,
    cudaStream_t stream)
{
    const int active_rows=sequence_length+target_length;
    const int elements=active_rows*embedding_dim;
    if (elements<=0 || target_length<=0 ||
        target_length>sequence_length ||
        num_codebooks<=0 || num_codebooks>8) {
        return;
    }
    constexpr int threads=256;
    int blocks=(elements+threads-1)/threads;
    if (blocks>4096) blocks=4096;
    omnivoice_prepare_cfg_embeddings_bf16_kernel
        <<<blocks,threads,0,stream>>>(
            tokens,embedding,codebook_offsets,base_input,out,
            num_codebooks,token_stride,sequence_length,target_length,
            embedding_dim,elements);
}

void omnivoice_argmax_scores_f32(
    const float* log_probs,
    const float* layer_penalty,
    int64_t* predictions,
    float* scores,
    int rows,
    int cols,
    int target_length,
    int mask_col,
    cudaStream_t stream)
{
    if (rows<=0 || cols<=0 || target_length<=0) return;
    omnivoice_argmax_scores_f32_kernel<<<rows,256,0,stream>>>(
        log_probs,layer_penalty,predictions,scores,
        cols,target_length,mask_col);
}

void omnivoice_maskgit_topk_update(
    const float* scores,
    const int64_t* predictions,
    int64_t* tokens,
    int elements,
    int k,
    int64_t mask_id,
    cudaStream_t stream)
{
    if (elements<=0 || elements>kSelectLargeCapacity || k<=0) return;
    launch_maskgit_topk<kSelectSmallItems,kSelectLargeItems>(
        elements,stream,
            scores,predictions,tokens,elements,k,
            nullptr,nullptr,0,mask_id);
}

void omnivoice_maskgit_topk_gumbel_update(
    const float* scores,
    const float* uniforms,
    const int64_t* predictions,
    int64_t* tokens,
    int elements,
    int k,
    int64_t mask_id,
    float temperature,
    cudaStream_t stream)
{
    if (elements<=0 || elements>kSelectLargeCapacity ||
        k<=0 || temperature<=0.0f) {
        return;
    }
    launch_maskgit_topk_gumbel<
        kSelectSmallItems,kSelectLargeItems>(
            elements,stream,
            scores,uniforms,predictions,tokens,
            elements,k,nullptr,nullptr,0,mask_id,temperature);
}

void omnivoice_maskgit_topk_scheduled_update(
    const float* scores,
    const int64_t* predictions,
    int64_t* tokens,
    const int* schedule,
    int* step,
    int elements,
    int schedule_length,
    int64_t mask_id,
    cudaStream_t stream)
{
    if (elements<=0 || elements>kSelectLargeCapacity ||
        schedule==nullptr || step==nullptr || schedule_length<=0) {
        return;
    }
    launch_maskgit_topk<kSelectSmallItems,kSelectLargeItems>(
        elements,stream,
            scores,predictions,tokens,elements,0,
            schedule,step,schedule_length,mask_id);
}

void omnivoice_maskgit_topk_gumbel_scheduled_update(
    const float* scores,
    const float* uniforms,
    const int64_t* predictions,
    int64_t* tokens,
    const int* schedule,
    int* step,
    int elements,
    int schedule_length,
    int64_t mask_id,
    float temperature,
    cudaStream_t stream)
{
    if (elements<=0 || elements>kSelectLargeCapacity ||
        schedule==nullptr || step==nullptr || schedule_length<=0 ||
        temperature<=0.0f) {
        return;
    }
    launch_maskgit_topk_gumbel<
        kSelectSmallItems,kSelectLargeItems>(
            elements,stream,
            scores,uniforms,predictions,tokens,elements,0,
            schedule,step,schedule_length,mask_id,temperature);
}
