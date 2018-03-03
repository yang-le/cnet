#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

void cublas_init(void);
void cublas_deinit(void);
cublasHandle_t cublas_handle(void);