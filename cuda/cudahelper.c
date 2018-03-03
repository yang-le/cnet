#include "cudahelper.h"

static cublasHandle_t handle = 0;

void cublas_init(void)
{
    cublasCreate(&handle);
}

void cublas_deinit(void)
{
    cublasDestroy(handle);
}

cublasHandle_t cublas_handle(void)
{
    return handle;
}
