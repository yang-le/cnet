#include "data.h"
#include <math.h>

void data_update_adam(data_t *data)
{
#define ALPHA 0.001
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8

    static int t = 0;
    int j = 0;

    ++t;

    for (j = 0; j < data->size; ++j)
    {
        data->m[j] = BETA1 * data->m[j] + (1 - BETA1) * data->grad[j];
        data->v[j] = BETA2 * data->v[j] + (1 - BETA2) * data->grad[j] * data->grad[j];
        data->val[j] -= ALPHA * sqrt(1 - pow(BETA2, t)) / (1 - pow(BETA1, t)) * data->m[j] / (sqrt(data->v[j]) + EPSILON);
        data->grad[j] = 0;
    }
}

void data_update(data_t *data, double rate)
{
    int j = 0;

    if (!data->immutable)
        for (j = 0; j < data->size; ++j)
        {
            data->val[j] -= rate * data->grad[j];
            data->grad[j] = 0;
        }
}

#ifdef USE_OPENCL
void cl_data_map(cl_data_val_t *data, int size)
{
    data->p = clEnqueueMapBuffer(cl_get_queues(0, 0), data->buf, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size * sizeof(data_val_t), 0, NULL, NULL, NULL);
}

void cl_data_unmap(cl_data_val_t *data)
{
    clEnqueueUnmapMemObject(cl_get_queues(0, 0), data->buf, data->p, 0, NULL, NULL);
}
#endif

size_t data_init(data_t *data, data_val_t *buf, int level)
{
    data_val_t *start = buf;

    data->val = buf;
    data->grad = data->val;
    buf += data->size;
#ifdef USE_CUDA
    cudaMalloc(&data->cuval, data->size * sizeof(data_val_t));

    data->cugrad = data->cuval;
#elif defined(USE_OPENCL)
    data->clval.buf = clCreateBuffer(cl_get_context(0), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, data->size * sizeof(data_val_t), data->val, NULL);
    cl_data_map(&data->clval, data->size);

    data->clgrad.buf = data->clval.buf;
#endif

    if (level > 0)
    {
        data->grad = buf;
        buf += data->size;
#ifdef USE_CUDA
        cudaMalloc(&data->cugrad, data->size * sizeof(data_val_t));
#elif defined(USE_OPENCL)
        data->clgrad.buf = clCreateBuffer(cl_get_context(0), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, data->size * sizeof(data_val_t), data->grad, NULL);
        cl_data_map(&data->clgrad, data->size);
#endif
    }

    if (level > 1)
    {
        data->m = buf;
        buf += data->size;
    }

    if (level > 2)
    {
        data->v = buf;
        buf += data->size;
    }

    return buf - start;
}

void data_load(FILE *fp, data_t *data)
{
    if (!data->immutable && data->size)
    {
        fread(data->val, sizeof(data_val_t), data->size, fp);
    }
}

void data_save(const data_t *data, FILE *fp)
{
    if (!data->immutable && data->size)
    {
        fwrite(data->val, sizeof(data_val_t), data->size, fp);
    }
}
