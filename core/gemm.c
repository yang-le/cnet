#include "gemm.h"
#if defined(USE_BLAS)
#include <cblas.h>
#else
// see https://github.com/pjreddie/darknet/blob/master/src/gemm.c

static void gemm_nn(int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float *C, int ldc)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < M; ++i)
    {
        int j, k;
        for (k = 0; k < K; ++k)
        {
            float A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j)
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

static void gemm_nt(int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float *C, int ldc)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < M; ++i)
    {
        int j, k;
        for (j = 0; j < N; ++j)
        {
            float sum = 0;
            for (k = 0; k < K; ++k)
            {
                sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}

static void gemm_tn(int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float *C, int ldc)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < M; ++i)
    {
        int j, k;
        for (k = 0; k < K; ++k)
        {
            float A_PART = ALPHA * A[k * lda + i];
            for (j = 0; j < N; ++j)
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

static void gemm_tt(int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float *C, int ldc)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < M; ++i)
    {
        int j, k;
        for (j = 0; j < N; ++j)
        {
            float sum = 0;
            for (k = 0; k < K; ++k)
            {
                sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}
#endif

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
          data_val_t **A, int offa, int lda,
          data_val_t **B, int offb, int ldb,
          float BETA,
          data_val_t **C, int offc, int ldc)
{
#if defined(USE_BLAS)
    cblas_sgemm(
        CblasRowMajor,
        TA ? CblasTrans : CblasNoTrans,
        TB ? CblasTrans : CblasNoTrans,
        M, N, K, ALPHA, *A + offa, lda, *B + offb, ldb, BETA, *C + offc, ldc);
#else
    int i, j;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            (*C + offc)[i * ldc + j] *= BETA;
        }
    }
    if (!TA && !TB)
        gemm_nn(M, N, K, ALPHA, *A + offa, lda, *B + offb, ldb, *C + offc, ldc);
    else if (TA && !TB)
        gemm_tn(M, N, K, ALPHA, *A + offa, lda, *B + offb, ldb, *C + offc, ldc);
    else if (!TA && TB)
        gemm_nt(M, N, K, ALPHA, *A + offa, lda, *B + offb, ldb, *C + offc, ldc);
    else
        gemm_tt(M, N, K, ALPHA, *A + offa, lda, *B + offb, ldb, *C + offc, ldc);
#endif
}
