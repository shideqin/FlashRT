// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — Pad FP4 packed + SFA between different shape layouts.

#include "pad_fp4_sfa.cuh"

#include <cstdint>
#include <cuda_runtime.h>

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(__CUDA_ARCH__)
#  include "cutlass/cutlass.h"
#  include "cutlass/detail/sm100_blockscaled_layout.hpp"
#  include "cute/tensor.hpp"
#  define FV_HAVE_CUTLASS 1
#else
#  define FV_HAVE_CUTLASS 0
#endif

namespace flash_rt {
namespace fused_fp4 {

#if FV_HAVE_CUTLASS

using CfgF4 = cutlass::detail::Sm1xxBlockScaledConfig<16>;

// Kernel: each thread handles one row's worth of packed data.
// Packed data is row-major (no layout dependency).
__global__ void pad_fp4_packed_kernel(
    const uint8_t* __restrict__ src_packed,
    uint8_t* __restrict__ dst_packed,
    int src_rows, int dst_rows, int D_bytes) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dst_rows) return;

    if (row < src_rows) {
        // Copy valid row
        const uint8_t* src = src_packed + row * D_bytes;
        uint8_t* dst = dst_packed + row * D_bytes;
        for (int i = threadIdx.y; i < D_bytes; i += blockDim.y) {
            dst[i] = src[i];
        }
    } else {
        // Zero-fill padding row
        uint8_t* dst = dst_packed + row * D_bytes;
        for (int i = threadIdx.y; i < D_bytes; i += blockDim.y) {
            dst[i] = 0;
        }
    }
}

template <class LayoutSrc, class LayoutDst>
__global__ void pad_fp4_sfa_kernel(
    const uint8_t* __restrict__ src_sfa,
    uint8_t* __restrict__ dst_sfa,
    LayoutSrc layout_src, LayoutDst layout_dst,
    int src_rows, int dst_rows, int D) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_blocks = D / 16;
    const int block_idx = blockIdx.y;
    if (row >= dst_rows || block_idx >= n_blocks) return;

    const int col_base = block_idx * 16;
    int dst_off = layout_dst(row, col_base, 0);

    if (row < src_rows) {
        int src_off = layout_src(row, col_base, 0);
        dst_sfa[dst_off] = src_sfa[src_off];
    } else {
        dst_sfa[dst_off] = 0;  // Zero scale → zero contribution
    }
}

#endif  // FV_HAVE_CUTLASS

void pad_fp4_sfa(
    const uint8_t* src_packed, const uint8_t* src_sfa,
    uint8_t* dst_packed, uint8_t* dst_sfa,
    int src_rows, int dst_rows, int D,
    cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    const int D_bytes = D / 2;  // FP4: 2 elements per byte

    // Copy packed data (1D grid over rows, 2D block for intra-row parallelism)
    {
        const int threads_per_row = 32;
        const int rows_per_block = 4;
        dim3 block(rows_per_block, threads_per_row);
        dim3 grid((dst_rows + rows_per_block - 1) / rows_per_block);
        pad_fp4_packed_kernel<<<grid, block, 0, stream>>>(
            src_packed, dst_packed, src_rows, dst_rows, D_bytes);
    }

    // Remap SFA (2D grid: rows × blocks)
    {
        auto shape_src = cute::make_shape(src_rows, 1, D, 1);
        auto shape_dst = cute::make_shape(dst_rows, 1, D, 1);
        auto layout_src = CfgF4::tile_atom_to_shape_SFA(shape_src);
        auto layout_dst = CfgF4::tile_atom_to_shape_SFA(shape_dst);

        const int n_blocks = D / 16;
        const int block_rows = 8;
        dim3 block_dim(block_rows);
        dim3 grid_dim((dst_rows + block_rows - 1) / block_rows, n_blocks);
        pad_fp4_sfa_kernel<<<grid_dim, block_dim, 0, stream>>>(
            src_sfa, dst_sfa, layout_src, layout_dst,
            src_rows, dst_rows, D);
    }
#else
    (void)src_packed; (void)src_sfa; (void)dst_packed; (void)dst_sfa;
    (void)src_rows; (void)dst_rows; (void)D; (void)stream;
#endif
}

}  // namespace fused_fp4
}  // namespace flash_rt
