#pragma once

#include "data.h"

#ifdef __cplusplus
extern "C" {
#endif

void im2col(data_val_t** data_im, int off_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, data_val_t** data_col, int off_col);

void col2im(data_val_t** data_col, int off_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, data_val_t** data_im, int off_im);

#ifdef __cplusplus
}
#endif
