
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16

namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
    

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    (void)W_out; // silence declared but never referenced warning. remove this line when you start working

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int b = blockDim.x * blockIdx.x + threadIdx.x;

    if (b < B) // for each image in the batch
    {
        for (int m = 0; m < M; m++)             // for each output feature maps
            for (int h = 0; h < H_out; h++)     // for each output element (height)
                for (int w = 0; w < W_out; w++) // for each output element (width)
                {
                    y4d(b, m, h, w) = 0;
                    for (int c = 0; c < C; c++)     // sum over all input feature maps
                        for (int p = 0; p < K; p++) // KxK filter
                            for (int q = 0; q < K; q++)
                                y4d(b, m, h, w) += x4d(b, c, h + p, w + q) * k4d(m, c, p, q); // No halo elements. Discard edges.
                }
    }

#undef y4d
#undef x4d
#undef k4d
}

__global__ void tiled_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, int W_grid)
{
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    extern __shared__ float shared[];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int b = blockIdx.x;
    int m = blockIdx.y;
    int edge = (K - 1) / 2;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int h_tile_idx = blockIdx.z / W_grid;
    int w_tile_idx = blockIdx.z % W_grid;

    int row_o = h_tile_idx * TILE_WIDTH + ty;
    int col_o = w_tile_idx * TILE_WIDTH + tx;
    int row_i = row_o - edge;
    int col_i = col_o - edge;

    if(b < B) {
        float output = 0;
        for(int c = 0; c < C; c++) { // sum over all input features

            // Load shared memory
            if(row_i >= 0 && row_i < H && col_i >= 0 && col_i < W) {
                shared[ty * (TILE_WIDTH+K-1) + tx] = x4d(b, c, row_i, col_i);
            } else {
                shared[ty * (TILE_WIDTH+K-1) + tx] = 0;
            }
            __syncthreads();

            // Calculate convolution
            if(ty < TILE_WIDTH && tx < TILE_WIDTH) { // Only a fraction of thread will participate in calculation
                for(int i = 0; i < K; i++) {
                    for(int j = 0; j < K; j++) {
                        output += shared[(i + ty) * (TILE_WIDTH+K-1) + tx + j] * k4d(m, c, i, j);
                    }
                }
            }
            __syncthreads();
        }

        // Only inner part. Output tile is smaller
        if(row_o >= edge && row_o < H - edge && col_o >= edge && col_o < W - edge) {
            y4d(b, m, row_o - edge, col_o - edge) = output;
        }
        __syncthreads();

    }


    #undef y4d
    #undef x4d
    #undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // // Use mxnet's CHECK_EQ to do assertions.
    // // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    const int B = x.shape_[0]; // Number of images in the batch
    const int M = y.shape_[1]; // Number of output feature maps
    const int C = x.shape_[1]; // Number of input feature maps
    const int H = x.shape_[2]; // Height of each input map image
    const int W = x.shape_[3]; // Width of each input map image
    const int K = w.shape_[3]; // Height and width of each filter bank

    // dim3 gridDim((B + 511) / 512);
    // dim3 blockDim(512);

    // MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    // forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
    // MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    // from chapter 16
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_grid = ceil(W/(float)TILE_WIDTH); // number of horizontal tiles per output map
    int H_grid = ceil(H/(float)TILE_WIDTH); // number of vertical tiles per output map
    int Z = H_grid * W_grid;       // number of tiles
    dim3 blockDim(TILE_WIDTH + K - 1, TILE_WIDTH + K - 1, 1);
    dim3 gridDim(B, M, Z);
    size_t shmem_size = sizeof(float) * ( (TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) );
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    tiled_forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K, W_grid);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    assert(0 && "No forward implementation for other datatypes needed for ECE408");
}
}
}

#endif