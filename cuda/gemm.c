#include "cudahelper.h"

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
    data_val_t** A, int offa, int lda,
    data_val_t** B, int offb, int ldb,
    float BETA,
    data_val_t** C, int offc, int ldc)
{
    data_val_t** cuA = A + 1;
    data_val_t** cuB = B + 1;
    data_val_t** cuC = C + 1;

    cublasSetVector(M * K, sizeof(data_val_t), *A + offa, 1, *cuA + offa, 1);
    cublasSetVector(K * N, sizeof(data_val_t), *B + offb, 1, *cuB + offb, 1);
    cublasSetVector(M * N, sizeof(data_val_t), *C + offc, 1, *cuC + offc, 1);

    cublasSgemm(
        cublas_handle(),
        TB ? CUBLAS_OP_T : CUBLAS_OP_N,
        TA ? CUBLAS_OP_T : CUBLAS_OP_N,
        N, M, K, &ALPHA, *cuB + offb, ldb, *cuA + offa, lda, &BETA, *cuC + offc, ldc);

    cublasGetVector(M * N, sizeof(data_val_t), *cuC + offc, 1, *C + offc, 1);
}
