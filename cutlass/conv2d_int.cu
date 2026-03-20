// Author: Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


// Modified by Kiwan Maeng for CrypTorch project.
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/tensor_view_io.h>
#include <cutlass/numeric_types.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/kernel/default_conv2d_dgrad.h>
#include <cutlass/conv/kernel/default_conv2d_wgrad.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/util/host_tensor.h>
#include "conv2d_int.h"

#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <random>

#include <cublas_v2.h>

const int block_size = 256;

template <typename T>
using Conv2DFprop = typename cutlass::conv::kernel::DefaultConv2dFprop<
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T, // accumulator, might be overkill for small bitwidths but that's okay
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm70,
    //cutlass::gemm::GemmShape<64, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        T,
        1,
        T,
        T>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
>;
template <typename T>
using Conv2DKernel = typename Conv2DFprop<T>::Kernel;


// typedef typename Conv2DFprop<int_t>::IteratorA::does_not_exist test;

/*
using Conv2DGroupKernel = typename cutlass::conv::kernel::DefaultConv2dGroupFprop<
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T, // accumulator, might be overkill for small bitwidths but that's okay
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm86,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        T,
        1,
        T,
        T>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::GroupMode::kDepthWise,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;
*/

template <typename T>
using Conv2DImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2DKernel<T>>;

//template <typename T>
//using Conv2DImplicitGemmGroup = cutlass::conv::device::ImplicitGemmConvolution<Conv2DGroupKernel<T>>;

void cutlass_conv2d64(GPUConv2DKey<int64_t> k, int64_t *d_I, int64_t *d_F, int64_t *d_C)
{
    // auto start = std::chrono::high_resolution_clock::now();
    auto A = cutlass::TensorRef<int64_t, cutlass::layout::TensorNHWC>(
        d_I,
        cutlass::layout::TensorNHWC::packed({k.p.N, k.p.H, k.p.W, k.p.CI}));
    auto B = cutlass::TensorRef<int64_t, cutlass::layout::TensorNHWC>(
        d_F,
        cutlass::layout::TensorNHWC::packed({k.p.CO, k.p.FH, k.p.FW, k.p.CI}));
    auto C = cutlass::TensorRef<int64_t, cutlass::layout::TensorNHWC>(
        d_C,
        cutlass::layout::TensorNHWC::packed({k.p.N, k.p.OH, k.p.OW, k.p.CO}));
    int64_t alpha = 1;
    int64_t beta = 0;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    cutlass::Tensor4DCoord input_size(k.p.N, k.p.H, k.p.W, k.p.CI);
    cutlass::Tensor4DCoord filter_size(k.p.CO, k.p.FH, k.p.FW, k.p.CI);
    // check these initializations later
    cutlass::Tensor4DCoord padding(k.p.zPadHLeft, k.p.zPadHRight, k.p.zPadWLeft, k.p.zPadWRight);
    cutlass::MatrixCoord conv_stride(k.p.strideH, k.p.strideW);
    cutlass::MatrixCoord dilation(1, 1);
    cutlass::Tensor4DCoord output_size(k.p.N, k.p.OH, k.p.OW, k.p.CO);

    cutlass::conv::Conv2dProblemSize problem_size(
        input_size,
        filter_size,
        padding,
        conv_stride,
        dilation,
        output_size,
        mode,
        1 // split_k_slices
    );

    cudaError_t result;
    typename Conv2DImplicitGemm<int64_t>::Arguments arguments{
        problem_size,
        A,
        B,
        C,
        C,
        {alpha, beta}
    };

    Conv2DImplicitGemm<int64_t> implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
    // printf("Allocating gpu workspace\n");

    uint8_t *workspace;
    cudaMallocAsync(&workspace, workspace_size, 0);

    // printf("Allocation done\n");
    // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    auto status = implicit_gemm_op.can_implement(arguments);
    //CUTLASS_CHECK(status);

    status = implicit_gemm_op.initialize(arguments, workspace);
    //CUTLASS_CHECK(status);

    status = implicit_gemm_op();
    //CUTLASS_CHECK(status);

    cudaFreeAsync(workspace, 0);

    cudaDeviceSynchronize();

    result = cudaGetLastError();

    if (result != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: "
          << cudaGetErrorString(result) << std::endl;
    }
}

template <typename T>
void cutlass_conv2d(Conv2DParams const &p, T *d_I, T *d_F, T *d_C)
{
    // auto start = std::chrono::high_resolution_clock::now();
    auto A = cutlass::TensorRef<T, cutlass::layout::TensorNHWC>(
        d_I,
        cutlass::layout::TensorNHWC::packed({p.N, p.H, p.W, p.CI}));
    auto B = cutlass::TensorRef<T, cutlass::layout::TensorNHWC>(
        d_F,
        cutlass::layout::TensorNHWC::packed({p.CO, p.FH, p.FW, p.CI}));
    auto C = cutlass::TensorRef<T, cutlass::layout::TensorNHWC>(
        d_C,
        cutlass::layout::TensorNHWC::packed({p.N, p.OH, p.OW, p.CO}));
    T alpha = 1;
    T beta = 0;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    cutlass::Tensor4DCoord input_size(p.N, p.H, p.W, p.CI);
    cutlass::Tensor4DCoord filter_size(p.CO, p.FH, p.FW, p.CI);
    // check these initializations later
    cutlass::Tensor4DCoord padding(p.zPadHLeft, p.zPadHRight, p.zPadWLeft, p.zPadWRight);
    cutlass::MatrixCoord conv_stride(p.strideH, p.strideW);
    cutlass::MatrixCoord dilation(1, 1);
    cutlass::Tensor4DCoord output_size(p.N, p.OH, p.OW, p.CO);

    cutlass::conv::Conv2dProblemSize problem_size(
        input_size,
        filter_size,
        padding,
        conv_stride,
        dilation,
        output_size,
        mode,
        1 // split_k_slices
    );

    cudaError_t result;
    typename Conv2DImplicitGemm<T>::Arguments arguments{
        problem_size,
        A,
        B,
        C,
        C,
        {alpha, beta}
    };

    Conv2DImplicitGemm<T> implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
    // printf("Allocating gpu workspace\n");

    uint8_t *workspace;
    cudaMallocAsync(&workspace, workspace_size, 0);

    // printf("Allocation done\n");
    // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    auto status = implicit_gemm_op.can_implement(arguments);
    //CUTLASS_CHECK(status);

    status = implicit_gemm_op.initialize(arguments, workspace);
    //CUTLASS_CHECK(status);

    status = implicit_gemm_op();
    //CUTLASS_CHECK(status);

    cudaFreeAsync(workspace, 0);

    cudaDeviceSynchronize();

    result = cudaGetLastError();

    if (result != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: "
          << cudaGetErrorString(result) << std::endl;
    }
}

void cutlass_conv2d64(Conv2DParams const &p, int64_t *d_I, int64_t *d_F, int64_t *d_C) {
    cutlass_conv2d<int64_t>(p, d_I, d_F, d_C);
}
void cutlass_conv2d32(Conv2DParams const &p, int32_t *d_I, int32_t *d_F, int32_t *d_C) {
    cutlass_conv2d<int32_t>(p, d_I, d_F, d_C);
}
// void cutlass_conv2d16(Conv2DParams const &p, int16_t *d_I, int16_t *d_F, int16_t *d_C) {
//     cutlass_conv2d<int16_t>(p, d_I, d_F, d_C);
// }