#include <torch/extension.h>
#include <ATen/ATen.h>
#include "gemm_int.h"
#include <iostream>

void matmul64(torch::Tensor &A, torch::Tensor &B, torch::Tensor &C, int M, int K, int N, int BS=1)
{
    bool isBatched = false;
    if (B.strides().size() == 3)
        isBatched = true;
    bool isRowMajor = false;
    if (B.strides()[1] == 1) {
        // B is row-major
        isRowMajor = true;
    }
    cutlass_gemm64(A.data_ptr<int64_t>(), B.data_ptr<int64_t>(), C.data_ptr<int64_t>(), M, K, N, isRowMajor, isBatched, BS);
}

void matmul32(torch::Tensor &A, torch::Tensor &B, torch::Tensor &C, int M, int K, int N, int BS=1)
{
    bool isBatched = false;
    if (B.strides().size() == 3)
        isBatched = true;
    bool isRowMajor = false;
    if (B.strides()[1] == 1) {
        // B is row-major
        isRowMajor = true;
    }
    cutlass_gemm32(A.data_ptr<int32_t>(), B.data_ptr<int32_t>(), C.data_ptr<int32_t>(), M, K, N, isRowMajor, isBatched, BS);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul64", &matmul64);
  m.def("matmul32", &matmul32);
}
