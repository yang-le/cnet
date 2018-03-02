#pragma once

#include "cublasXt.h"

void cublas_init(void);
void cublas_deinit(void);
cublasXtHandle_t cublas_handle(void);