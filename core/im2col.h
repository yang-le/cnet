#pragma once

#include "data.h"

void im2col(data_val_t** data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, data_val_t** data_col);

void col2im(data_val_t** data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, data_val_t** data_im);
