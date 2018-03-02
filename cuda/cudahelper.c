#include "cudahelper.h"

static cublasXtHandle_t handle = 0;

void cublas_init(void)
{
    int deviceId[1] = {0};
    cublasXtCreate(&handle);
    cublasXtDeviceSelect(handle, 1, deviceId);
}

void cublas_deinit(void)
{
    cublasXtDestroy(handle);
}

cublasXtHandle_t cublas_handle(void)
{
    return handle;
}
