#include "im2col.h"

//see https://github.com/pjreddie/darknet/blob/master/src/im2col.c

static float im2col_get_pixel(float *im, int height, int width, int channels,
                              int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width)
        return 0;
    return im[col + width * (row + height * channel)];
}

void im2col(data_val_t **data_im, int off_im,
            int channels, int height, int width,
            int ksize, int stride, int pad, data_val_t **data_col, int off_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c)
    {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h)
        {
            for (w = 0; w < width_col; ++w)
            {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                (*data_col + off_col)[col_index] = im2col_get_pixel(*data_im + off_im, height, width, channels,
                                                                    im_row, im_col, c_im, pad);
            }
        }
    }
}

//see https://github.com/pjreddie/darknet/blob/master/src/col2im.c

static void col2im_add_pixel(float *im, int height, int width, int channels,
                             int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width)
        return;
    im[col + width * (row + height * channel)] += val;
}

void col2im(data_val_t **data_col, int off_col,
            int channels, int height, int width,
            int ksize, int stride, int pad, data_val_t **data_im, int off_im)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c)
    {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h)
        {
            for (w = 0; w < width_col; ++w)
            {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = (*data_col + off_col)[col_index];
                col2im_add_pixel(*data_im + off_im, height, width, channels,
                                 im_row, im_col, c_im, pad, val);
            }
        }
    }
}
