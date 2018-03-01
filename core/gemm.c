#ifdef USE_OPENCL
#include "clhelper.h"
#endif
#if defined(USE_CLBLAST)
#include "clblast_c.h"
#elif defined(USE_CLBLAS)
#include "clBLAS.h"
#elif defined(USE_BLAS)
#include <cblas.h>
#endif
#include "gemm.h"

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

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
          data_val_t **A, int lda,
          data_val_t **B, int ldb,
          float BETA,
          data_val_t **C, int ldc)
{
#if defined(USE_OPENCL)
    cl_event event = NULL;
    cl_command_queue queue = cl_get_queues(0, 0);

    cl_data_val_t *clA = (cl_data_val_t *)(A + 1);
    cl_data_val_t *clB = (cl_data_val_t *)(B + 1);
    cl_data_val_t *clC = (cl_data_val_t *)(C + 1);

    cl_data_unmap(clA);
    cl_data_unmap(clB);
    cl_data_unmap(clC);
#if defined(USE_CLBLAST)
    CLBlastSgemm(
        CLBlastLayoutRowMajor,
        TA ? CLBlastTransposeYes : CLBlastTransposeNo,
        TB ? CLBlastTransposeYes : CLBlastTransposeNo,
        M, N, K, ALPHA, clA->buf, 0, lda, clB->buf, 0, ldb, BETA, clC->buf, 0, ldc,
        &queue, &event
    );
#elif defined(USE_CLBLAS)
    clblasSgemm(
        clblasRowMajor,
        TA ? clblasTrans : clblasNoTrans,
        TB ? clblasTrans : clblasNoTrans,
        M, N, K, ALPHA, clA->buf, 0, lda, clB->buf, 0, ldb, BETA, clC->buf, 0, ldc,
        1, &queue, 0, NULL, &event
    );
#endif
    clWaitForEvents(1, &event);

    cl_data_map(clA, M * K);
    cl_data_map(clB, K * N);
    cl_data_map(clC, M * N);
#elif defined(USE_BLAS)
    cblas_sgemm(
        CblasRowMajor,
        TA ? CblasTrans : CblasNoTrans,
        TB ? CblasTrans : CblasNoTrans,
        M, N, K, ALPHA, *A, lda, *B, ldb, BETA, *C, ldc
    );
#else
    int i, j;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            (*C)[i * ldc + j] *= BETA;
        }
    }
    if (!TA && !TB)
        gemm_nn(M, N, K, ALPHA, *A, lda, *B, ldb, *C, ldc);
    else if (TA && !TB)
        gemm_tn(M, N, K, ALPHA, *A, lda, *B, ldb, *C, ldc);
    else if (!TA && TB)
        gemm_nt(M, N, K, ALPHA, *A, lda, *B, ldb, *C, ldc);
    else
        gemm_tt(M, N, K, ALPHA, *A, lda, *B, ldb, *C, ldc);
#endif
}
