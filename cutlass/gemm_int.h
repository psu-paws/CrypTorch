#ifndef __GEMMINT__
#define __GEMMINT__

void cutlass_gemm64(int64_t *A, int64_t *B, int64_t *C, int M, int K, int N, bool isRowMajor, bool isBatched, int BS);
void cutlass_gemm32(int32_t *A, int32_t *B, int32_t *C, int M, int K, int N, bool isRowMajor, bool isBatched, int BS);

#endif
