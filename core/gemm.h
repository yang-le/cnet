#pragma once

#include "data.h"

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
          data_val_t **A, int lda,
          data_val_t **B, int ldb,
          float BETA,
          data_val_t **C, int ldc);
