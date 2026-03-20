#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/tensor_view_io.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/util/host_tensor.h>

#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <random>

#include <cublas_v2.h>

template<typename T>
using CutlassBatchedGemmRRR = cutlass::gemm::device::GemmBatched<T,
                                                cutlass::layout::RowMajor,
                                                T,
                                                cutlass::layout::RowMajor,
                                                T,
                                                cutlass::layout::RowMajor>;


template<typename T>
using CutlassGemmRRR = cutlass::gemm::device::Gemm<T,
                                                cutlass::layout::RowMajor,
                                                T,
                                                cutlass::layout::RowMajor,
                                                T,
                                                cutlass::layout::RowMajor>;

template<typename T>
using CutlassGemmRCR = cutlass::gemm::device::Gemm<T,
                                                cutlass::layout::RowMajor,
                                                T,
                                                cutlass::layout::ColumnMajor,
                                                T,
                                                cutlass::layout::RowMajor>;

template<typename T>
cudaError_t CutlassGemmCallRCR(
  int M,
  int K,
  int N,
  T alpha,
  T const *A,
  int lda,
  T const *B,
  int ldb,
  T beta,
  T *C,
  int ldc) {

  CutlassGemmRCR<T> gemm_operator;


  typename CutlassGemmRCR<T>::Arguments args({M, N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta});

  cutlass::Status status = gemm_operator(args);

  if (status != cutlass::Status::kSuccess) {
      std::cout << "Error" << std::endl;
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

template<typename T>
cudaError_t CutlassBatchedGemmCallRRR(
  int M,
  int K,
  int N,
  T alpha,
  T const *A,
  int lda,
  T const *B,
  int ldb,
  T beta,
  T *C,
  int ldc,
  int batch_stride_A,
  int batch_stride_B,
  int batch_stride_C,
  int batch_count) {

  CutlassBatchedGemmRRR<T> gemm_operator;


  typename CutlassBatchedGemmRRR<T>::Arguments args({M, N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              batch_stride_A,
                              {B, ldb},    // Tensor-ref for source matrix B
                              batch_stride_B,
                              {C, ldc},    // Tensor-ref for source matrix C
                              batch_stride_C,
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              batch_stride_C,
                              {alpha, beta},
                              batch_count);

  cutlass::Status status = gemm_operator(args);

  if (status != cutlass::Status::kSuccess) {
      std::cout << "Error" << std::endl;
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}


template<typename T>
cudaError_t CutlassGemmCallRRR(
  int M,
  int K,
  int N,
  T alpha,
  T const *A,
  int lda,
  T const *B,
  int ldb,
  T beta,
  T *C,
  int ldc) {

  CutlassGemmRRR<T> gemm_operator;


  typename CutlassGemmRRR<T>::Arguments args({M, N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta});

  cutlass::Status status = gemm_operator(args);

  if (status != cutlass::Status::kSuccess) {
      std::cout << "Error" << std::endl;
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

//void CutlassGemm64(uint64_t *A, uint64_t *B, uint64_t *C, int M, int K, int N, bool isRowMajor, bool isBatched, int BS) {
void cutlass_gemm64(int64_t *A, int64_t *B, int64_t *C, int M, int K, int N, bool isRowMajor, bool isBatched, int BS) {
    int64_t alpha = 1;
    int64_t beta = 0;

    cudaError_t result;

    //int lda = M;
    //int ldb = K;
    //int ldc = N;
    int lda = K;
    int ldb;
    // TOOD: Batched is always assumed to be row-major.
    if (isRowMajor or isBatched)
        ldb = N;
    else
        ldb = K;
    int ldc = N;

    if (isBatched)
        CutlassBatchedGemmCallRRR<int64_t>(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc, M*K, K*N, M*N, BS);
        //CutlassBatchedGemmCallRRR<uint64_t>(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc, M*K, K*N, M*N, BS);
    else if (isRowMajor)
        CutlassGemmCallRRR<int64_t>(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc);
        //CutlassGemmCallRRR<uint64_t>(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        CutlassGemmCallRCR<int64_t>(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc);
        //CutlassGemmCallRCR<uint64_t>(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc);
    cudaDeviceSynchronize();

    result = cudaGetLastError();

    if (result != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: "
          << cudaGetErrorString(result) << std::endl;
    }
}

void cutlass_gemm32(int32_t *A, int32_t *B, int32_t *C, int M, int K, int N, bool isRowMajor, bool isBatched, int BS) {
    int32_t alpha = 1;
    int32_t beta = 0;

    cudaError_t result;

    //int lda = M;
    //int ldb = K;
    //int ldc = N;
    int lda = K;
    int ldb;
    // TOOD: Batched is always assumed to be row-major.
    if (isRowMajor or isBatched)
        ldb = N;
    else
        ldb = K;
    int ldc = N;

    if (isBatched)
        CutlassBatchedGemmCallRRR<int32_t>(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc, M*K, K*N, M*N, BS);
        //CutlassBatchedGemmCallRRR<uint32_t>(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc, M*K, K*N, M*N, BS);
    else if (isRowMajor)
        CutlassGemmCallRRR<int32_t>(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc);
        //CutlassGemmCallRRR<uint32_t>(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        CutlassGemmCallRCR<int32_t>(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc);
        //CutlassGemmCallRCR<uint32_t>(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc);
    cudaDeviceSynchronize();

    result = cudaGetLastError();

    if (result != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: "
          << cudaGetErrorString(result) << std::endl;
    }
}
