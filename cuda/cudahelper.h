#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

void cublas_init(void);
void cublas_deinit(void);
cublasHandle_t cublas_handle(void);

#ifdef __cplusplus
}
#endif
