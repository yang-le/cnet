#pragma once

void im2col(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

void col2im(float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_im);
