#include <cuda_runtime_api.h>
#include <stdio.h>
#include "kernel.h"


#define ELEMENTS_PER_THREADS 8

#define DIVUP(x, y) (((x) + (y)-1) / (y))

template <typename T>
__host__ __device__ T saturate(const T x, const T lower, const T upper) {
    if (x < lower)
        return lower;
    if (x > upper)
        return upper;
    return x;
}

__host__ __device__ inline void interpolate_cubic_coefs(float x,
                                                        float* coeffs) {
    const float A = -0.75f;
    coeffs[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    coeffs[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    coeffs[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

template <size_t CH>
__global__ void resize_cubic_32f_kernel_vector(
        const float* __restrict__ src, float* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const float row_scale,
        const float col_scale) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y * ELEMENTS_PER_THREADS + threadIdx.y;
    if (dr < dst_rows && dc < dst_cols) {
        float fc = ((float)dc + 0.5) * col_scale - 0.5;
        int sc = floor(fc);
        fc -= sc;
        float coef_col[4];
        interpolate_cubic_coefs(fc, coef_col);

        for (int i = 0; i < ELEMENTS_PER_THREADS; i++) {
            if (dr >= dst_rows)
                return;
            float fr = ((float)dr + 0.5) * row_scale - 0.5;
            int sr = floor(fr);
            fr -= sr;
            float coef_row[4];
            interpolate_cubic_coefs(fr, coef_row);
            float dst_data[CH] = {0};
#pragma unroll
            for (int offset_r = 0; offset_r < 4; ++offset_r) {
                int tr_step =
                        saturate(sr + offset_r - 1, 0, (int)src_rows - 1) *
                        src_step;
#pragma unroll
                for (int offset_c = 0; offset_c < 4; ++offset_c) {
                    int tc_step =
                            saturate(sc + offset_c - 1, 0, (int)src_cols - 1) *
                            CH;
                    int src_address = tr_step + tc_step;
#pragma unroll
                    for (size_t ch = 0; ch < CH; ++ch) {
                        dst_data[ch] += coef_row[offset_r] *
                                        coef_col[offset_c] * src[src_address++];
                    }
                }
            }
            int dst_address = dr * dst_step + dc * CH;
#pragma unroll
            for (int i = 0; i < CH; i++)
                dst[dst_address++] = dst_data[i];
            dr += blockDim.y;
        }
    }
}

__global__ void precompute_cubic_coef_f32(float* dst, float scale,
                                          size_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;

    float fr = (tid + 0.5) * scale - 0.5;
    int* sr = (int*)(dst + size * 4);
    sr[tid] = (int)(floorf(fr));

    fr -= sr[tid];
    float coef[4];
    interpolate_cubic_coefs(fr, coef);
#pragma unroll
    for (int j = 0, index = 0; j < 4; j++, index += size) {
        dst[tid + index] = coef[j];
    }
}

template <size_t CH>
__global__ void resize_cubic_32f_kernel_cacheToGlobal(
        const float* src, float* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const float* gl_coef_row,
        const float* gl_coef_col, const int* gl_sr, const int* gl_sc) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y * ELEMENTS_PER_THREADS + threadIdx.y;

    if (dr < dst_rows && dc < dst_cols) {
        int sc = gl_sc[dc];
        float coef_col[4];
#pragma unroll
        for (int i = 0, index = dc; i < 4; i++, index += dst_cols)
            coef_col[i] = gl_coef_col[index];

        for (int i = 0; i < ELEMENTS_PER_THREADS; i++) {
            if (dr >= dst_rows)
                return;
            int sr = gl_sr[dr];
            float coef_row[4];
#pragma unroll
            for (int i = 0, index = dr; i < 4; i++, index += dst_rows)
                coef_row[i] = gl_coef_row[index];

            float dst_data[CH] = {0};
#pragma unroll
            for (int offset_r = 0; offset_r < 4; ++offset_r) {
                int tr_step =
                        saturate(sr + offset_r - 1, 0, (int)src_rows - 1) *
                        src_step;
#pragma unroll
                for (int offset_c = 0; offset_c < 4; ++offset_c) {
                    int tc_step =
                            saturate(sc + offset_c - 1, 0, (int)src_cols - 1) *
                            CH;
                    int src_address = tr_step + tc_step;
#pragma unroll
                    for (size_t ch = 0; ch < CH; ++ch) {
                        dst_data[ch] += coef_row[offset_r] *
                                        coef_col[offset_c] * src[src_address++];
                    }
                }
            }
            int dst_address = dr * dst_step + dc * CH;
#pragma unroll
            for (int i = 0; i < CH; i++)
                dst[dst_address++] = dst_data[i];

            dr += blockDim.y;
        }
    }
}


void resize_cubic(const float* src, float* dst, const size_t src_rows,
                       const size_t src_cols, const size_t dst_rows,
                       const size_t dst_cols, const size_t src_step,
                       const size_t dst_step, void* workspace,
                       cudaStream_t stream) {
    dim3 THREADS(32, 8, 1);
    float row_scale = (float)src_rows / dst_rows;
    float col_scale = (float)src_cols / dst_cols;

    size_t dst_area_size = dst_rows * dst_cols;
    size_t src_area_size = src_rows * src_cols;

    bool enlarge = dst_area_size > src_area_size;
    bool shrink = dst_area_size <= src_area_size;

    bool use_vector = (enlarge && (dst_area_size <= 500 * 500)) ||
                      (shrink && (dst_area_size <= 1000 * 1000));

    if (use_vector) {
        dim3 BLOCKS(DIVUP(dst_cols, THREADS.x),
                    DIVUP(dst_rows, THREADS.y * ELEMENTS_PER_THREADS));

        resize_cubic_32f_kernel_vector<1><<<BLOCKS, THREADS, 0, stream>>>(
                src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step,
                dst_step, row_scale, col_scale);
    } else {
        dim3 BLOCKS(DIVUP(dst_cols, THREADS.x),
                    DIVUP(dst_rows, THREADS.y * ELEMENTS_PER_THREADS));

        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        float* dev_coef_row = static_cast<float*>(workspace);
        int* dev_sr = reinterpret_cast<int*>(dev_coef_row + dst_rows * 4);
        float* dev_coef_col = reinterpret_cast<float*>(dev_sr + dst_rows);
        int* dev_sc = reinterpret_cast<int*>(dev_coef_col + dst_cols * 4);

        precompute_cubic_coef_f32<<<DIVUP(dst_rows, 128), 128, 0, stream>>>(
                dev_coef_row, row_scale, dst_rows);
        precompute_cubic_coef_f32<<<DIVUP(dst_cols, 128), 128, 0, stream>>>(
                dev_coef_col, col_scale, dst_cols);

        resize_cubic_32f_kernel_cacheToGlobal<1>
                <<<BLOCKS, THREADS, 0, stream>>>(
                        (const float*)src, (float*)dst, src_rows, src_cols,
                        dst_rows, dst_cols, src_step, dst_step, dev_coef_row,
                        dev_coef_col, dev_sr, dev_sc);
    }
}

