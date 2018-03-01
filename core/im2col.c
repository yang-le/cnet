#ifdef USE_CLBLAST
#include "clhelper.h"
#include "clblast_c.h"
#endif
#include "im2col.h"

//see https://github.com/pjreddie/darknet/blob/master/src/im2col.c

static float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

void im2col(data_val_t** data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, data_val_t** data_col) 
{
    int channels_col = channels * ksize * ksize;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    #ifdef USE_CLBLAST
    cl_event event = NULL;
    cl_command_queue queue = cl_get_queues(0, 0);

    cl_data_val_t *clim = (cl_data_val_t *)(data_im + 1);
    cl_data_val_t *clcol = (cl_data_val_t *)(data_col + 1);

    cl_data_unmap(clim);
    cl_data_unmap(clcol);

    CLBlastSim2col(
        channels, height, width,
        ksize, ksize, pad, pad, stride, stride, 0, 0,
        clim->buf, 0, clcol->buf, 0,
        &queue, &event
    );
    clWaitForEvents(1, &event);

    cl_data_map(clim, channels * height * width);
    cl_data_map(clcol, channels_col * height_col * height_col);
    #else
    int c,h,w;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                (*data_col)[col_index] = im2col_get_pixel(*data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
    #endif
}

//see https://github.com/pjreddie/darknet/blob/master/src/col2im.c

static void col2im_add_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;
    im[col + width*(row + height*channel)] += val;
}

void col2im(data_val_t** data_col,
         int channels,  int height,  int width,
         int ksize,  int stride, int pad, data_val_t** data_im) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = (*data_col)[col_index];
                col2im_add_pixel(*data_im, height, width, channels,
                        im_row, im_col, c_im, pad, val);
            }
        }
    }
}
