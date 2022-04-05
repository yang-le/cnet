#include "data.h"
#include "log.h"
#include <math.h>

void data_update_nesterov(data_t *data)
{
    const double ALPHA = 0.9;

    int j = 0;

    for (j = 0; j < data->size; ++j)
    {
        data->val[j] += ALPHA * data->m[j];
    }
}

void data_update_sgd(data_t *data, double rate)
{
    int j = 0;

    for (j = 0; j < data->size; ++j)
    {
        data->val[j] -= rate * data->grad[j];
        data->grad[j] = 0;
    }
}

void data_update_momentum(data_t *data, double rate)
{
    const double ALPHA = 0.9;

    int j = 0;

    for (j = 0; j < data->size; ++j)
    {
        data->m[j] = ALPHA * data->m[j] - rate * data->grad[j];
        data->val[j] += data->m[j];
        data->grad[j] = 0;
    }
}

void data_update_adagrad(data_t *data, double rate)
{
    const double EPSILON = 1e-7;

    int j = 0;

    for (j = 0; j < data->size; ++j)
    {
        data->m[j] += data->grad[j] * data->grad[j];
        data->val[j] -= rate * data->grad[j] / (sqrt(data->m[j]) + EPSILON);
        data->grad[j] = 0;
    }
}

void data_update_adadelta(data_t *data, double rate)
{
    const double BETA = 0.5;
    const double EPSILON = 1e-6;

    int j = 0;

    for (j = 0; j < data->size; ++j)
    {
        data->m[j] = BETA * data->m[j] + (1 - BETA) * data->grad[j] * data->grad[j];
        data->val[j] -= rate * data->grad[j] / sqrt(data->m[j] + EPSILON);
        data->grad[j] = 0;
    }
}

void data_update_adam(data_t *data, double rate, int t)
{
    const double BETA1 = 0.9;
    const double BETA2 = 0.999;
    const double EPSILON = 1e-8;

    int j = 0;

    for (j = 0; j < data->size; ++j)
    {
        data->m[j] = BETA1 * data->m[j] + (1 - BETA1) * data->grad[j];
        data->v[j] = BETA2 * data->v[j] + (1 - BETA2) * data->grad[j] * data->grad[j];
        data->val[j] -= rate * sqrt(1 - pow(BETA2, t)) / (1 - pow(BETA1, t)) * data->m[j] / (sqrt(data->v[j]) + EPSILON);
        data->grad[j] = 0;
    }
}

#ifdef USE_OPENCL
void cl_data_map(cl_data_val_t *data)
{
    cl_int clRet = 0;
    data->p = clEnqueueMapBuffer(cl_get_default_queues(), data->buf, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, data->size, 0, NULL, NULL, &clRet);
    if (clRet != CL_SUCCESS)
    {
        LOG("fail to map buffer in opencl, you can try with a smaller batch size.\n");
        exit(1);
    }
}

void cl_data_unmap(cl_data_val_t *data)
{
    cl_int clRet = clEnqueueUnmapMemObject(cl_get_default_queues(), data->buf, data->p, 0, NULL, NULL);
    if (CL_SUCCESS != clRet)
    {
        LOG("fail to unmap buffer in opencl, you can try with a smaller batch size.\n");
        exit(1);
    }
}
#endif

size_t data_init(data_t *data, data_val_t *buf, int level, int batch)
{
    data_val_t *start = buf;
    size_t data_size = data->size * batch;

    if (0 == data_size)
        return 0;

    data->val = buf;
    data->grad = data->val;
    buf += data_size;
#ifdef USE_CUDA
    if (cudaSuccess != cudaMalloc((void **)&data->cuval, data_size * sizeof(data_val_t)))
    {
        LOG("fail to alloc %ld bytes memory in cuda, you can try with a smaller batch size.\n", data_size * sizeof(data_val_t));
        exit(1);
    }

    data->cugrad = data->cuval;
#elif defined(USE_OPENCL)
    cl_int clRet = 0;
    data->clval.size = data_size * sizeof(data_val_t);
    data->clval.buf = clCreateBuffer(cl_get_default_context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, data->clval.size, data->val, &clRet);
    if (clRet != CL_SUCCESS)
    {
        LOG("fail to alloc %d bytes memory in opencl, you can try with a smaller batch size.\n", data->clval.size);
        exit(1);
    }

    cl_data_map(&data->clval);
    data->clgrad = data->clval;
#endif

    if (level > 0)
    {
        data->grad = buf;
        buf += data_size;
#ifdef USE_CUDA
        if (cudaSuccess != cudaMalloc((void **)&data->cugrad, data_size * sizeof(data_val_t)))
        {
            LOG("fail to alloc %ld bytes memory in cuda, you can try with a smaller batch size.\n", data_size * sizeof(data_val_t));
            exit(1);
        }
#elif defined(USE_OPENCL)
        cl_int clRet = 0;
        data->clgrad.size = data_size * sizeof(data_val_t);
        data->clgrad.buf = clCreateBuffer(cl_get_default_context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, data->clgrad.size, data->grad, &clRet);
        if (clRet != CL_SUCCESS)
        {
            LOG("fail to alloc %d bytes memory in opencl, you can try with a smaller batch size.\n", data->clgrad.size);
            exit(1);
        }

        cl_data_map(&data->clgrad);
#endif
    }

    if (level > 1)
    {
        data->m = buf;
        buf += data_size;
    }

    if (level > 2)
    {
        data->v = buf;
        buf += data_size;
    }

    return buf - start;
}

void data_load(FILE *fp, data_t *data)
{
    if (data->size)
    {
        fread(data->val, sizeof(data_val_t), data->size, fp);
    }
}

void data_save(const data_t *data, FILE *fp)
{
    if (data->size)
    {
        fwrite(data->val, sizeof(data_val_t), data->size, fp);
    }
}
#ifdef USE_OPENCV
void cv_data_show(char *window, int delay, data_t *data, CvMat *cvdata, int offset, int iw, int ih, int ow, int oh)
{
    CvMat temp;

    if (!cvGetWindowHandle(window))
    {
        cvNamedWindow(window, CV_WINDOW_AUTOSIZE);
    }

    cvInitMatHeader(&temp, ih, iw, CV_32FC1, data->val + offset, CV_AUTOSTEP);

    if ((iw == ow) && (ih == oh))
    {
        cvShowImage(window, &temp);
    }
    else
    {
        if ((!cvdata) || (cvdata->cols != ow) || (cvdata->rows != oh))
        {
            cvReleaseMat(&cvdata);
            cvdata = cvCreateMat(oh, ow, CV_32FC1);
        }

        cvResize(&temp, cvdata, CV_INTER_LINEAR);
        cvShowImage(window, cvdata);
    }
    cvWaitKey(delay);
}
#endif
