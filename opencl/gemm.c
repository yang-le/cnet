#include "gemm.h"
#include "clhelper.h"
#include "clBLAS.h"

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
    data_val_t** A, int offa, int lda,
    data_val_t** B, int offb, int ldb,
    float BETA,
    data_val_t** C, int offc, int ldc)
{
    cl_event event = NULL;
    cl_command_queue queue = cl_get_default_queues();

    cl_data_val_t* clA = (cl_data_val_t*)(A + 1);
    cl_data_val_t* clB = (cl_data_val_t*)(B + 1);
    cl_data_val_t* clC = (cl_data_val_t*)(C + 1);

    cl_data_unmap(clA);
    cl_data_unmap(clB);
    cl_data_unmap(clC);

    clblasSgemm(
        clblasRowMajor,
        TA ? clblasTrans : clblasNoTrans,
        TB ? clblasTrans : clblasNoTrans,
        M, N, K, ALPHA, clA->buf, offa, lda, clB->buf, offb, ldb, BETA, clC->buf, offc, ldc,
        1, &queue, 0, NULL, &event);
    clWaitForEvents(1, &event);

    cl_data_map(clA);
    cl_data_map(clB);
    cl_data_map(clC);
}
