// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — Fused Q/K RMSNorm + RoPE kernel.
//
// Single-kernel warp-per-head Q/K normalization + RoPE rotation.
// Each warp processes one attention head independently using shuffle-based
// RMS reduction — no shared memory, no __syncthreads() within a head.
// Replaces qkv_split + rms_norm + rope_apply (3 kernel launches → 1).
//
// Pipeline per warp:
//   1. Load head elements into registers from strided Dq buffer
//   2. Compute RMS sum via warp-shuffle butterfly reduction
//   3. Normalize with per-head Q/K weight, apply RoPE from registers
//   4. Write RoPE result to flat output buffer

#include "omnivoice_qk_norm_rope.cuh"
#include "common.cuh"

namespace flash_rt {
namespace kernels {

__global__ void omnivoice_rms_apply_exact_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ inv_rms,
    __nv_bfloat16* __restrict__ out,
    int dim)
{
    int row=blockIdx.x;
    float scale=inv_rms[row];
    for (int col=threadIdx.x;col<dim;col+=blockDim.x) {
        int idx=row*dim+col;
        float norm_bf=to_f32(from_f32<__nv_bfloat16>(
            to_f32(x[idx])*scale));
        out[idx]=from_f32<__nv_bfloat16>(
            norm_bf*to_f32(weight[col]));
    }
}

void omnivoice_rms_apply_exact_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const float* inv_rms,
    __nv_bfloat16* out,
    int rows, int dim,
    cudaStream_t stream)
{
    omnivoice_rms_apply_exact_kernel<<<rows,256,0,stream>>>(
        x,weight,inv_rms,out,dim);
}

__global__ void omnivoice_rms_apply_from_mean_exact_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ mean_square,
    __nv_bfloat16* __restrict__ out,
    int dim,
    float eps)
{
    int row=blockIdx.x;
    float scale=rsqrtf(__fadd_rn(mean_square[row],eps));
    for (int col=threadIdx.x;col<dim;col+=blockDim.x) {
        int idx=row*dim+col;
        float norm_bf=to_f32(from_f32<__nv_bfloat16>(
            to_f32(x[idx])*scale));
        out[idx]=from_f32<__nv_bfloat16>(
            norm_bf*to_f32(weight[col]));
    }
}

void omnivoice_rms_apply_from_mean_exact_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const float* mean_square,
    __nv_bfloat16* out,
    int rows,
    int dim,
    float eps,
    cudaStream_t stream)
{
    omnivoice_rms_apply_from_mean_exact_kernel<<<rows,256,0,stream>>>(
        x,weight,mean_square,out,dim,eps);
}

__global__ void omnivoice_square_bf16_f32_kernel(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ out,
    int elements)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    for (;idx<elements;idx+=blockDim.x*gridDim.x) {
        float value=to_f32(x[idx]);
        out[idx]=__fmul_rn(value,value);
    }
}

void omnivoice_square_bf16_f32(
    const __nv_bfloat16* x,
    float* out,
    int elements,
    cudaStream_t stream)
{
    if (elements <= 0) return;
    int blocks=(elements+255)/256;
    if (blocks>4096) blocks=4096;
    omnivoice_square_bf16_f32_kernel<<<blocks,256,0,stream>>>(
        x,out,elements);
}

__global__ void omnivoice_residual_add_square_bf16_f32_kernel(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ update,
    float* __restrict__ square,
    int elements)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    for (;idx<elements;idx+=blockDim.x*gridDim.x) {
        // Preserve PyTorch's two-stage semantics exactly: the residual add
        // first rounds back to BF16, then RMSNorm promotes that stored BF16
        // value to FP32 before squaring.
        __nv_bfloat16 summed=from_f32<__nv_bfloat16>(
            to_f32(residual[idx])+to_f32(update[idx]));
        residual[idx]=summed;
        float value=to_f32(summed);
        square[idx]=__fmul_rn(value,value);
    }
}

void omnivoice_residual_add_square_bf16_f32(
    __nv_bfloat16* residual,
    const __nv_bfloat16* update,
    float* square,
    int elements,
    cudaStream_t stream)
{
    if (elements <= 0) return;
    int blocks=(elements+255)/256;
    if (blocks>4096) blocks=4096;
    omnivoice_residual_add_square_bf16_f32_kernel<<<blocks,256,0,stream>>>(
        residual,update,square,elements);
}

template <bool AddResidual>
__global__ void omnivoice_square_mean_1024_bf16_f32_kernel(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ update,
    float* __restrict__ mean_square,
    int rows)
{
    constexpr int kWarpsPerBlock=8;
    int lane=threadIdx.x&31;
    int row=blockIdx.x*kWarpsPerBlock+(threadIdx.x>>5);
    if (row>=rows) return;

    __nv_bfloat16* row_residual=residual+row*1024;
    const __nv_bfloat16* row_update=
        AddResidual ? update+row*1024 : nullptr;

    // Match ATen's contiguous 1024-element mean reduction: four independent
    // accumulators per lane, eight values per accumulator, then combine the
    // accumulators in order before the descending warp-shuffle reduction.
    // TensorIterator selects the scalar load layout for a single output and
    // the vectorized-input layout when multiple rows are reduced.
    float acc0=0.0f;
    float acc1=0.0f;
    float acc2=0.0f;
    float acc3=0.0f;
    #pragma unroll
    for (int group=0;group<8;++group) {
        int base=(rows==1 ? lane : lane*4)+group*128;
        int component_stride=(rows==1 ? 32 : 1);
        __nv_bfloat16 v0=row_residual[base];
        __nv_bfloat16 v1=row_residual[base+component_stride];
        __nv_bfloat16 v2=row_residual[base+2*component_stride];
        __nv_bfloat16 v3=row_residual[base+3*component_stride];
        if constexpr (AddResidual) {
            v0=from_f32<__nv_bfloat16>(
                to_f32(v0)+to_f32(row_update[base]));
            v1=from_f32<__nv_bfloat16>(
                to_f32(v1)+to_f32(row_update[base+component_stride]));
            v2=from_f32<__nv_bfloat16>(
                to_f32(v2)+to_f32(row_update[base+2*component_stride]));
            v3=from_f32<__nv_bfloat16>(
                to_f32(v3)+to_f32(row_update[base+3*component_stride]));
            row_residual[base]=v0;
            row_residual[base+component_stride]=v1;
            row_residual[base+2*component_stride]=v2;
            row_residual[base+3*component_stride]=v3;
        }
        float f0=to_f32(v0);
        float f1=to_f32(v1);
        float f2=to_f32(v2);
        float f3=to_f32(v3);
        acc0=__fadd_rn(acc0,__fmul_rn(f0,f0));
        acc1=__fadd_rn(acc1,__fmul_rn(f1,f1));
        acc2=__fadd_rn(acc2,__fmul_rn(f2,f2));
        acc3=__fadd_rn(acc3,__fmul_rn(f3,f3));
    }
    float sum=__fadd_rn(acc0,acc1);
    sum=__fadd_rn(sum,acc2);
    sum=__fadd_rn(sum,acc3);
    #pragma unroll
    for (int offset=16;offset>0;offset>>=1) {
        sum=__fadd_rn(sum,__shfl_down_sync(0xffffffffu,sum,offset));
    }
    if (lane==0) {
        mean_square[row]=__fmul_rn(sum,1.0f/1024.0f);
    }
}

void omnivoice_square_mean_1024_bf16_f32(
    const __nv_bfloat16* x,
    float* mean_square,
    int rows,
    cudaStream_t stream)
{
    if (rows<=0) return;
    int blocks=(rows+7)/8;
    omnivoice_square_mean_1024_bf16_f32_kernel<false>
        <<<blocks,256,0,stream>>>(
            const_cast<__nv_bfloat16*>(x),nullptr,mean_square,rows);
}

void omnivoice_residual_add_square_mean_1024_bf16_f32(
    __nv_bfloat16* residual,
    const __nv_bfloat16* update,
    float* mean_square,
    int rows,
    cudaStream_t stream)
{
    if (rows<=0) return;
    int blocks=(rows+7)/8;
    omnivoice_square_mean_1024_bf16_f32_kernel<true>
        <<<blocks,256,0,stream>>>(
            residual,update,mean_square,rows);
}

template <bool AddResidual>
__global__ void omnivoice_rms_norm_exact_1024_bf16_kernel(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ update,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int rows,
    float eps)
{
    int row=blockIdx.x;
    if (row>=rows) return;
    __shared__ float shared_square[1024];
    __nv_bfloat16* row_residual=residual+row*1024;
    const __nv_bfloat16* row_update=
        AddResidual ? update+row*1024 : nullptr;

    #pragma unroll
    for (int col=threadIdx.x;col<1024;col+=blockDim.x) {
        __nv_bfloat16 value=row_residual[col];
        if constexpr (AddResidual) {
            value=from_f32<__nv_bfloat16>(
                to_f32(value)+to_f32(row_update[col]));
            row_residual[col]=value;
        }
        float value_f32=to_f32(value);
        shared_square[col]=__fmul_rn(value_f32,value_f32);
    }
    __syncthreads();

    if (threadIdx.x<32) {
        int lane=threadIdx.x;
        float acc0=0.0f;
        float acc1=0.0f;
        float acc2=0.0f;
        float acc3=0.0f;
        #pragma unroll
        for (int group=0;group<8;++group) {
            int base=(rows==1 ? lane : lane*4)+group*128;
            int component_stride=(rows==1 ? 32 : 1);
            acc0=__fadd_rn(acc0,shared_square[base]);
            acc1=__fadd_rn(
                acc1,shared_square[base+component_stride]);
            acc2=__fadd_rn(
                acc2,shared_square[base+2*component_stride]);
            acc3=__fadd_rn(
                acc3,shared_square[base+3*component_stride]);
        }
        float sum=__fadd_rn(acc0,acc1);
        sum=__fadd_rn(sum,acc2);
        sum=__fadd_rn(sum,acc3);
        #pragma unroll
        for (int offset=16;offset>0;offset>>=1) {
            sum=__fadd_rn(
                sum,__shfl_down_sync(0xffffffffu,sum,offset));
        }
        if (lane==0) {
            float mean=__fmul_rn(sum,1.0f/1024.0f);
            shared_square[0]=rsqrtf(__fadd_rn(mean,eps));
        }
    }
    __syncthreads();
    float scale=shared_square[0];
    for (int col=threadIdx.x;col<1024;col+=blockDim.x) {
        float norm_bf=to_f32(from_f32<__nv_bfloat16>(
            to_f32(row_residual[col])*scale));
        out[row*1024+col]=from_f32<__nv_bfloat16>(
            norm_bf*to_f32(weight[col]));
    }
}

void omnivoice_rms_norm_exact_1024_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    int rows,
    float eps,
    cudaStream_t stream)
{
    if (rows<=0) return;
    omnivoice_rms_norm_exact_1024_bf16_kernel<false>
        <<<rows,256,0,stream>>>(
            const_cast<__nv_bfloat16*>(x),nullptr,weight,out,rows,eps);
}

void omnivoice_residual_add_rms_norm_exact_1024_bf16(
    __nv_bfloat16* residual,
    const __nv_bfloat16* update,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    int rows,
    float eps,
    cudaStream_t stream)
{
    if (rows<=0) return;
    omnivoice_rms_norm_exact_1024_bf16_kernel<true>
        <<<rows,256,0,stream>>>(
            residual,update,weight,out,rows,eps);
}

__device__ __forceinline__ float omnivoice_silu_f32(float x)
{
    return x/(1.0f+expf(-x));
}

__global__ void omnivoice_silu_mul_merged_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int dim)
{
    int row=blockIdx.x;
    int col=blockIdx.y*blockDim.x+threadIdx.x;
    int col_stride=gridDim.y*blockDim.x;
    for (;col<dim;col+=col_stride) {
        int idx=row*dim+col;
        int base=row*(2*dim)+col;
        float gate=to_f32(gate_up[base]);
        float up=to_f32(gate_up[base+dim]);
        // F.silu(BF16) rounds once before the following BF16 multiply.
        float silu_rounded=to_f32(from_f32<__nv_bfloat16>(
            omnivoice_silu_f32(gate)));
        out[idx]=from_f32<__nv_bfloat16>(silu_rounded*up);
    }
}

void omnivoice_silu_mul_merged_bf16(
    const __nv_bfloat16* gate_up,
    __nv_bfloat16* out,
    int rows,
    int dim,
    cudaStream_t stream)
{
    if (rows<=0 || dim<=0) return;
    int shards=(dim+255)/256;
    if (shards>1) shards=1;
    dim3 grid(rows,shards);
    omnivoice_silu_mul_merged_bf16_kernel<<<grid,256,0,stream>>>(
        gate_up,out,rows,dim);
}

__global__ void omnivoice_expand_kv_bf16_kernel(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ k_expanded,
    __nv_bfloat16* __restrict__ v_expanded,
    int batch,
    int seq,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int v_row_stride)
{
    int elements=batch*num_heads*seq*head_dim;
    int groups=num_heads/num_kv_heads;
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    for (;idx<elements;idx+=blockDim.x*gridDim.x) {
        int dim=idx%head_dim;
        int tmp=idx/head_dim;
        int pos=tmp%seq;
        tmp/=seq;
        int head=tmp%num_heads;
        int batch_idx=tmp/num_heads;
        int kv_head=head/groups;
        int row=batch_idx*seq+pos;
        int head_offset=kv_head*head_dim+dim;
        int k_src=row*num_kv_heads*head_dim+head_offset;
        int v_src=row*v_row_stride+head_offset;
        k_expanded[idx]=k[k_src];
        v_expanded[idx]=v[v_src];
    }
}

__global__ void omnivoice_expand_kv_128_bf16_kernel(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ k_expanded,
    __nv_bfloat16* __restrict__ v_expanded,
    int batch,
    int seq,
    int num_heads,
    int num_kv_heads,
    int v_row_stride)
{
    constexpr int kWarpsPerBlock=8;
    constexpr int kHeadDim=128;
    int lane=threadIdx.x&31;
    int output_row=blockIdx.x*kWarpsPerBlock+(threadIdx.x>>5);
    int output_rows=batch*num_heads*seq;
    if (output_row>=output_rows) return;

    int pos=output_row%seq;
    int tmp=output_row/seq;
    int head=tmp%num_heads;
    int batch_idx=tmp/num_heads;
    int kv_head=head/(num_heads/num_kv_heads);
    int input_row=batch_idx*seq+pos;
    int col=lane*4;

    const __nv_bfloat16* k_src=
        k+(input_row*num_kv_heads+kv_head)*kHeadDim+col;
    const __nv_bfloat16* v_src=
        v+input_row*v_row_stride+kv_head*kHeadDim+col;
    __nv_bfloat16* k_dst=
        k_expanded+output_row*kHeadDim+col;
    __nv_bfloat16* v_dst=
        v_expanded+output_row*kHeadDim+col;
    *reinterpret_cast<uint2*>(k_dst)=
        *reinterpret_cast<const uint2*>(k_src);
    *reinterpret_cast<uint2*>(v_dst)=
        *reinterpret_cast<const uint2*>(v_src);
}

void omnivoice_expand_kv_bf16(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* k_expanded,
    __nv_bfloat16* v_expanded,
    int batch,
    int seq,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int v_row_stride,
    cudaStream_t stream)
{
    int elements=batch*num_heads*seq*head_dim;
    if (elements<=0) return;
    if (head_dim==128) {
        int output_rows=batch*num_heads*seq;
        int blocks=(output_rows+7)/8;
        omnivoice_expand_kv_128_bf16_kernel<<<blocks,256,0,stream>>>(
            k,v,k_expanded,v_expanded,batch,seq,num_heads,
            num_kv_heads,v_row_stride);
        return;
    }
    int blocks=(elements+255)/256;
    if (blocks>4096) blocks=4096;
    omnivoice_expand_kv_bf16_kernel<<<blocks,256,0,stream>>>(
        k,v,k_expanded,v_expanded,batch,seq,num_heads,
        num_kv_heads,head_dim,v_row_stride);
}

__global__ void omnivoice_qk_norm_rope_kernel(
    const __nv_bfloat16* __restrict__ dq,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_temp,
    __nv_bfloat16* __restrict__ k_temp,
    __nv_bfloat16* __restrict__ k_expanded,
    __nv_bfloat16* __restrict__ v_expanded,
    int B, int S,
    int NH, int NKV, int HD, int QKVD, bool packed_kv, float eps)
{
    using T = __nv_bfloat16;

    constexpr int kWarpCount = 8;
    constexpr int kWarpSize = 32;
    constexpr int kElemsPerThread = 128 / kWarpSize;  // 4 for HD=128
    constexpr int kHalf = 128 >> 1;                    // 64
    constexpr int kRotSlotOffset = kHalf / kWarpSize;  // 2

    int row = blockIdx.x;
    int warp_id = threadIdx.x / kWarpSize;
    int lane_id = threadIdx.x % kWarpSize;
    int NQK = NH * HD;
    int KVD = NKV * HD;

    const T* cos_row = cos + row * HD;
    const T* sin_row = sin + row * HD;
    float cos_vals[kElemsPerThread];
    float sin_vals[kElemsPerThread];
    #pragma unroll
    for (int e = 0; e < kElemsPerThread; ++e) {
        int col = lane_id + e * kWarpSize;
        cos_vals[e] = to_f32(cos_row[col]);
        sin_vals[e] = to_f32(sin_row[col]);
    }

    // Phase 1: Q heads (NH=16, 2 iterations of 8)
    const T* dq_q = dq + row * QKVD;
    T* q_temp_row = q_temp + row * NQK;

    for (int iter = 0; iter < 2; ++iter) {
        int h = iter * kWarpCount + warp_id;
        if (h >= NH) continue;

        const T* head_src = dq_q + h * HD;
        T* head_dst = q_temp_row + h * HD;

        float vals[kElemsPerThread];
        float local_sum = 0.0f;
        #pragma unroll
        for (int e = 0; e < kElemsPerThread; ++e) {
            int col = lane_id + e * kWarpSize;
            vals[e] = to_f32(head_src[col]);
            float reduce_val=to_f32(head_src[lane_id*kElemsPerThread+e]);
            float square=__fmul_rn(reduce_val,reduce_val);
            local_sum=__fadd_rn(local_sum,square);
        }

        float total = warp_reduce_sum(local_sum);
        total = __shfl_sync(0xffffffff, total, 0);
        float rms = rsqrtf(total / HD + eps);

        #pragma unroll
        for (int e = 0; e < kElemsPerThread; ++e) {
            int col = lane_id + e * kWarpSize;
            // Match Qwen3RMSNorm's two BF16 rounding points:
            //   weight * (hidden * rsqrt(var + eps)).to(input_dtype)
            float norm_bf = to_f32(from_f32<T>(vals[e] * rms));
            vals[e] = to_f32(from_f32<T>(
                norm_bf * to_f32(q_weight[col])));
        }

        #pragma unroll
        for (int e = 0; e < kElemsPerThread; ++e) {
            int col = lane_id + e * kWarpSize;
            float cv = cos_vals[e];
            float sv = sin_vals[e];
            float xv = vals[e];

            int rot_slot = (col < kHalf) ? (e + kRotSlotOffset) : (e - kRotSlotOffset);
            float rot_val = vals[rot_slot];
            if (col < kHalf) rot_val = -rot_val;

            // PyTorch evaluates the two BF16 multiplies before the BF16 add.
            float rot_cos_bf = to_f32(from_f32<T>(xv * cv));
            float rot_sin_bf = to_f32(from_f32<T>(rot_val * sv));
            head_dst[col] = from_f32<T>(rot_cos_bf + rot_sin_bf);
        }
    }

    // Phase 2: K heads (NKV=8, 1 iteration of 8)
    const T* dq_k = dq + row * QKVD + NQK;
    const T* dq_v = dq_k + KVD;
    T* k_temp_row = k_temp ? k_temp + row * KVD : nullptr;

    {
        int h = warp_id;
        if (h < NKV) {
            const T* head_src = dq_k + h * HD;
            T* head_dst = k_temp_row ? k_temp_row + h * HD : nullptr;

            float vals[kElemsPerThread];
            float local_sum = 0.0f;
            #pragma unroll
            for (int e = 0; e < kElemsPerThread; ++e) {
                int col = lane_id + e * kWarpSize;
                vals[e] = to_f32(head_src[col]);
                float reduce_val=to_f32(
                    head_src[lane_id*kElemsPerThread+e]);
                float square=__fmul_rn(reduce_val,reduce_val);
                local_sum=__fadd_rn(local_sum,square);
            }

            float total = warp_reduce_sum(local_sum);
            total = __shfl_sync(0xffffffff, total, 0);
            float rms = rsqrtf(total / HD + eps);

            #pragma unroll
            for (int e = 0; e < kElemsPerThread; ++e) {
                int col = lane_id + e * kWarpSize;
                float norm_bf = to_f32(from_f32<T>(vals[e] * rms));
                vals[e] = to_f32(from_f32<T>(
                    norm_bf * to_f32(k_weight[col])));
            }

            #pragma unroll
            for (int e = 0; e < kElemsPerThread; ++e) {
                int col = lane_id + e * kWarpSize;
                float cv = cos_vals[e];
                float sv = sin_vals[e];
                float xv = vals[e];

                int rot_slot = (col < kHalf) ? (e + kRotSlotOffset) : (e - kRotSlotOffset);
                float rot_val = vals[rot_slot];
                if (col < kHalf) rot_val = -rot_val;

                float rot_cos_bf = to_f32(from_f32<T>(xv * cv));
                float rot_sin_bf = to_f32(from_f32<T>(rot_val * sv));
                T k_out=from_f32<T>(rot_cos_bf + rot_sin_bf);
                if (head_dst) head_dst[col]=k_out;
                if (k_expanded) {
                    int batch_idx=row/S;
                    int pos=row-batch_idx*S;
                    int groups=NH/NKV;
                    #pragma unroll
                    for (int group=0;group<2;++group) {
                        if (group<groups) {
                            int out_head=h*groups+group;
                            int out_idx=packed_kv
                                ? (row*NH+out_head)*HD+col
                                : ((batch_idx*NH+out_head)*S+pos)*HD+col;
                            k_expanded[out_idx]=k_out;
                            v_expanded[out_idx]=dq_v[h*HD+col];
                        }
                    }
                }
            }
        }
    }
}

void omnivoice_qk_norm_rope_bf16(
    const __nv_bfloat16* dq,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_temp,
    __nv_bfloat16* k_temp,
    int BS, int NH, int NKV, int HD, int QKVD, float eps,
    cudaStream_t stream)
{
    if (BS <= 0 || NH <= 0 || NKV <= 0 || HD <= 0) return;

    constexpr int kThreads = 256;
    dim3 block(kThreads);
    dim3 grid(BS);

    omnivoice_qk_norm_rope_kernel<<<grid, block, 0, stream>>>(
        dq, q_weight, k_weight, cos, sin,
        q_temp, k_temp, nullptr, nullptr, 0, 0,
        NH, NKV, HD, QKVD, false, eps);
}

void omnivoice_qk_norm_rope_expand_kv_bf16(
    const __nv_bfloat16* dq,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_temp,
    __nv_bfloat16* k_expanded,
    __nv_bfloat16* v_expanded,
    int batch,
    int seq,
    int NH,
    int NKV,
    int HD,
    int QKVD,
    int active_rows,
    float eps,
    cudaStream_t stream)
{
    int BS=batch*seq;
    if (BS<=0 || NH<=0 || NKV<=0 || HD<=0) return;
    if (active_rows<=0 || active_rows>BS) active_rows=BS;
    omnivoice_qk_norm_rope_kernel<<<active_rows,256,0,stream>>>(
        dq,q_weight,k_weight,cos,sin,q_temp,nullptr,
        k_expanded,v_expanded,batch,seq,
        NH,NKV,HD,QKVD,false,eps);
}

void omnivoice_qk_norm_rope_expand_packed_kv_bf16(
    const __nv_bfloat16* dq,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_temp,
    __nv_bfloat16* k_expanded,
    __nv_bfloat16* v_expanded,
    int batch,
    int seq,
    int NH,
    int NKV,
    int HD,
    int QKVD,
    int active_rows,
    float eps,
    cudaStream_t stream)
{
    int BS=batch*seq;
    if (BS<=0 || NH<=0 || NKV<=0 || HD<=0) return;
    if (active_rows<=0 || active_rows>BS) active_rows=BS;
    omnivoice_qk_norm_rope_kernel<<<active_rows,256,0,stream>>>(
        dq,q_weight,k_weight,cos,sin,q_temp,nullptr,
        k_expanded,v_expanded,batch,seq,
        NH,NKV,HD,QKVD,true,eps);
}

}  // namespace kernels
}  // namespace flash_rt
