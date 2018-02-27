#include "data.h"
#include "cl.h"
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

size_t data_init(data_t *data, data_val_t *buf, int level)
{
    data_val_t *start = buf;

    data->val = buf;
    buf += data->size;
#ifdef USE_OPENCL
    data->val_cl = clCreateBuffer(cl_get_context(0), CL_MEM_READ_WRITE, data->size * sizeof(data_val_t), NULL, NULL);
#endif

    if (level > 0)
    {
        data->grad = buf;
        buf += data->size;
#ifdef USE_OPENCL
        data->grad_cl = clCreateBuffer(cl_get_context(0), CL_MEM_READ_WRITE, data->size * sizeof(data_val_t), NULL, NULL);
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
