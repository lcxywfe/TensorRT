#include <algorithm>
#include <cuda_fp16.h>
#include <stdio.h>
#include "kernel.h"

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

#define CUBLAS_CHECK(condition)                                                                 \
    do                                                                                          \
    {                                                                                           \
        cublasStatus_t status = condition;                                                      \
        if (status != CUBLAS_STATUS_SUCCESS)                                                    \
        {                                                                                       \
            printf("%s %d CUBLAS FAIL %s\n", __FILE__, __LINE__, cublasGetErrorString(status)); \
        }                                                                                       \
    } while (0)


inline int GET_BLOCKS(const int N)
{
    return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

#define CUDA_1D_KERNEL_LOOP(i, n)                                                                                      \
    for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); i += (blockDim.x * gridDim.x))


int get_greatest_divisor_below_bound(int n, int bound) {
  for (int k = bound; k > 1; --k) {
    if (n % k == 0) {
      return k;
    }
  }
  return 1;
}

template <typename T>
__device__ T bilinear_interpolate(const T* in, const int height, const int width, T h, T w);

template <>
__device__ float bilinear_interpolate<float>(const float* in, const int height, const int width, float h, float w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = in[h_low * width + w_low];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = in[h_low * width + w_high];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = in[h_high * width + w_low];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = in[h_high * width + w_high];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <>
__device__ __half bilinear_interpolate<__half>(const __half* in, const int height, const int width, __half h, __half w) {
  if (h <= __half(-1.0) || __int2half_rd(height) <= h || w <= __half(-1.0) || __int2half_rd(width) <= w) {
    return 0;
  }

  int h_low = __half2int_rd(h);
  int w_low = __half2int_rd(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  __half lh = h - __int2half_rd(h_low);
  __half lw = w - __int2half_rd(w_low);
  __half hh = __half(1.0) - lh, hw = __half(1.0) - lw;

  __half v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = in[h_low * width + w_low];
  __half v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = in[h_low * width + w_high];
  __half v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = in[h_high * width + w_low];
  __half v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = in[h_high * width + w_high];

  __half w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  __half val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename T>
__global__ void deformable_im2col_gpu_kernel(
    const int n,
    const T* input_ptr,
    const T* offset_ptr,
    const T* mask_ptr,
    const int height,
    const int width,
    const int weight_h,
    const int weight_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dil_h,
    const int dil_w,
    const int batch_sz,
    const int n_in_channels,
    const int n_offset_grps,
    const int out_h,
    const int out_w,
    T* columns_ptr);

template <>
__global__ void deformable_im2col_gpu_kernel<float>(
    const int n,
    const float* input_ptr,
    const float* offset_ptr,
    const float* mask_ptr,
    const int height,
    const int width,
    const int weight_h,
    const int weight_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dil_h,
    const int dil_w,
    const int batch_sz,
    const int n_in_channels,
    const int n_offset_grps,
    const int out_h,
    const int out_w,
    float* columns_ptr) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int out_x = index % out_w;
    const int out_y = (index / out_w) % out_h;
    const int out_b = (index / (out_w * out_h)) % batch_sz;
    const int in_c = index / (out_w * out_h * batch_sz);
    const int out_c = in_c * weight_h * weight_w;

    int c_per_offset_grp = n_in_channels / n_offset_grps;
    const int grp_idx = in_c / c_per_offset_grp;

    columns_ptr +=
        (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) +
         out_y * out_w + out_x);

    input_ptr +=
        (out_b * (n_in_channels * height * width) + in_c * (height * width));

    offset_ptr += (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w *
        out_h * out_w;

    mask_ptr += (out_b * n_offset_grps + grp_idx) * weight_h * weight_w *
        out_h * out_w;

    for (int i = 0; i < weight_h; ++i) {
      for (int j = 0; j < weight_w; ++j) {
        const int offset_idx = (i * weight_w + j);
        const float offset_h =
            offset_ptr[2 * offset_idx * (out_h * out_w) + out_y * out_w + out_x];
        const float offset_w = offset_ptr
            [(2 * offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];
        const float mask =
            mask_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];

        const float y = (out_y * stride_h - pad_h) + i * dil_h + offset_h;
        const float x = (out_x * stride_w - pad_w) + j * dil_w + offset_w;
        float var = bilinear_interpolate(input_ptr, height, width, y, x);
        *columns_ptr = var * mask;
        columns_ptr += batch_sz * out_h * out_w;
      }
    }
  }
}

template <>
__global__ void deformable_im2col_gpu_kernel<__half>(
    const int n,
    const __half* input_ptr,
    const __half* offset_ptr,
    const __half* mask_ptr,
    const int height,
    const int width,
    const int weight_h,
    const int weight_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dil_h,
    const int dil_w,
    const int batch_sz,
    const int n_in_channels,
    const int n_offset_grps,
    const int out_h,
    const int out_w,
    __half* columns_ptr) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int out_x = index % out_w;
    const int out_y = (index / out_w) % out_h;
    const int out_b = (index / (out_w * out_h)) % batch_sz;
    const int in_c = index / (out_w * out_h * batch_sz);
    const int out_c = in_c * weight_h * weight_w;

    int c_per_offset_grp = n_in_channels / n_offset_grps;
    const int grp_idx = in_c / c_per_offset_grp;

    columns_ptr +=
        (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) +
         out_y * out_w + out_x);

    input_ptr +=
        (out_b * (n_in_channels * height * width) + in_c * (height * width));

    offset_ptr += (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w *
        out_h * out_w;

    mask_ptr += (out_b * n_offset_grps + grp_idx) * weight_h * weight_w *
        out_h * out_w;

    for (int i = 0; i < weight_h; ++i) {
      for (int j = 0; j < weight_w; ++j) {
        const int offset_idx = (i * weight_w + j);
        const __half offset_h =
            offset_ptr[2 * offset_idx * (out_h * out_w) + out_y * out_w + out_x];
        const __half offset_w = offset_ptr
            [(2 * offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];
        const __half mask =
            mask_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];

        const __half y = __hadd(__int2half_rd((out_y * stride_h - pad_h) + i * dil_h), offset_h);
        const __half x = __hadd(__int2half_rd((out_x * stride_w - pad_w) + j * dil_w), offset_w);
        __half var = bilinear_interpolate(input_ptr, height, width, y, x);
        *columns_ptr = var * mask;
        columns_ptr += batch_sz * out_h * out_w;
      }
    }
  }
}


template <typename T>
static void deformable_im2col(
    const T* input,
    const T* data_offset,
    const T* mask,
    int n_in_channels,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dil_h,
    int dil_w,
    int out_h,
    int out_w,
    int parallel_imgs,
    int deformable_group,
    T* data_col) {
  int num_kernels = n_in_channels * out_h * out_w * parallel_imgs;

        deformable_im2col_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS>>>(
            num_kernels,
            input,
            data_offset,
            mask,
            height,
            width,
            weight_h,
            weight_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dil_h,
            dil_w,
            parallel_imgs,
            n_in_channels,
            deformable_group,
            out_h,
            out_w,
            data_col);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
  }
}

template <typename T>
__global__ void add_bias(T* input,
                         const T*bias,
                         int out_channels,
                         int batch_sz,
                         int spatial_size) {
    T bias_c = bias[blockIdx.x % out_channels];
    T* input_c = input + blockIdx.x * spatial_size;

    int offset = threadIdx.x;
    while (offset < spatial_size) {
        input_c[offset] += bias_c;
        offset += blockDim.x;
    }
}

template <typename scalar_t>
__global__ void transpose_and_bias(scalar_t* input,
                         scalar_t* output,
                         const scalar_t *bias,
                         int batch_sz,
                         int parallel_sz,
                         int out_channels,
                         int spatial_size) {
    scalar_t bias_this_channel = bias[blockIdx.x % out_channels];
    scalar_t *output_str = output + blockIdx.x * spatial_size;

    int group_id = blockIdx.x / (out_channels * parallel_sz);
    int group_offset = blockIdx.x % (out_channels * parallel_sz);
    int bucket_id = group_offset / out_channels;
    int bucket_offset = group_offset % out_channels;
    int input_channel_offset = group_id * (out_channels * parallel_sz) +
                               bucket_offset * parallel_sz +
                               bucket_id;
    scalar_t *input_str = input + input_channel_offset * spatial_size;

    int offset = threadIdx.x;
    while (offset < spatial_size) {
        output_str[offset] = input_str[offset] + bias_this_channel;
        offset += blockDim.x;
    }
}

template <typename T>
void gemm(cublasHandle_t handle,
          cublasOperation_t transa, cublasOperation_t transb,
          int m, int n, int k,
          const T *alpha,
          const T *A, int lda,
          const T *B, int ldb,
          const T *beta,
          T *C, int ldc);

template <>
void gemm<float>(cublasHandle_t handle,
          cublasOperation_t transa, cublasOperation_t transb,
          int m, int n, int k,
          const float *alpha,
          const float *A, int lda,
          const float *B, int ldb,
          const float *beta,
          float *C, int ldc) {
	CUBLAS_CHECK(cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}

template <>
void gemm<__half>(cublasHandle_t handle,
          cublasOperation_t transa, cublasOperation_t transb,
          int m, int n, int k,
          const __half *alpha,
          const __half *A, int lda,
          const __half *B, int ldb,
          const __half *beta,
          __half *C, int ldc) {
	CUBLAS_CHECK(cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}


template<typename T>
void DeformConv2d_forward(
    const T* input,
    const T* offset,
    const T* mask,
    const T* weights,
    const T* bias,
    T* outbuf,
    T* output,
    const int batch_sz, const int in_channels, const int in_h, const int in_w,
    const int out_channels, const int weight_h, const int weight_w,
    std::pair<int, int> stride,
    std::pair<int, int> pad,
    std::pair<int, int> dilation,
    int groups, int deformable_groups, int im2col_step,
    cudaStream_t stream,
    cublasHandle_t handle) {

  // int n_parallel_imgs = get_greatest_divisor_below_bound(batch_sz, im2col_step);
  int n_parallel_imgs = 1;
  im2col_step = 1;

  int stride_h = stride.first;
  int stride_w = stride.second;
  int pad_h = pad.first;
  int pad_w = pad.second;
  int dil_h = dilation.first;
  int dil_w = dilation.second;
  int ker_h = dil_h * (weight_h - 1) + 1;
  int ker_w = dil_w * (weight_w - 1) + 1;
  int out_h = ((in_h + 2 * pad_h - ker_h) / stride_h) + 1;
  int out_w = ((in_w + 2 * pad_w - ker_w) / stride_w) + 1;

  T* columns = outbuf + out_h * out_w * out_channels * batch_sz;
  for (int b = 0; b < batch_sz / n_parallel_imgs; b++) {
    deformable_im2col(
        input + b * in_channels * in_h * in_w,
        offset + b * 2 * deformable_groups * weight_h * weight_w * in_h * in_w,
        mask + b * deformable_groups * weight_h * weight_w * in_h * in_w,
        in_channels,
        in_h,
        in_w,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dil_h,
        dil_w,
        out_h,
        out_w,
        n_parallel_imgs,
        deformable_groups,
        columns);

    T onef = 1.0, zerof = 0.0;
    gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    	n_parallel_imgs*out_h*out_w,   // N
    	out_channels,                  // M
    	in_channels*weight_h*weight_w, // K
    	&onef,   // alpha
    	columns, // Bt
   		n_parallel_imgs*out_h*out_w,   // N
    	weights, // At
    	in_channels*weight_h*weight_w, // K
    	&zerof,  // beta
    	outbuf + b * out_channels * out_h * out_w, // Ct
    	n_parallel_imgs*out_h*out_w    // N
    );
    //CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, out_channels, n_parallel_imgs*out_h*out_w, in_channels*weight_h*weight_w, &onef,
    //            weights, in_channels*weight_h*weight_w,
    //            columns, n_parallel_imgs*out_h*out_w, &zerof,
    //            outbuf + b * out_channels * out_h * out_w, n_parallel_imgs *out_h*out_w));
  }
  // add_bias<<<batch_sz * out_channels, CUDA_NUM_THREADS>>>
  //             (output, bias, out_channels, batch_sz, out_h * out_w);
  transpose_and_bias<<<batch_sz * out_channels, CUDA_NUM_THREADS>>>
              (outbuf, output, bias, batch_sz, n_parallel_imgs, out_channels, out_h * out_w);
}

template void DeformConv2d_forward<float>(
    const float* input,
    const float* offset,
    const float* mask,
    const float* weights,
    const float* bias,
    float* columns,
    float* output,
    const int batch_sz, const int in_channels, const int in_h, const int in_w,
    const int out_channels, const int weight_h, const int weight_w,
    std::pair<int, int> stride,
    std::pair<int, int> pad,
    std::pair<int, int> dilation,
    int groups, int deformable_groups, int im2col_step,
    cudaStream_t stream,
    cublasHandle_t handle);

template void DeformConv2d_forward<__half>(
    const __half* input,
    const __half* offset,
    const __half* mask,
    const __half* weights,
    const __half* bias,
    __half* columns,
    __half* output,
    const int batch_sz, const int in_channels, const int in_h, const int in_w,
    const int out_channels, const int weight_h, const int weight_w,
    std::pair<int, int> stride,
    std::pair<int, int> pad,
    std::pair<int, int> dilation,
    int groups, int deformable_groups, int im2col_step,
    cudaStream_t stream,
    cublasHandle_t handle);

