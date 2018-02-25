#include "data.h"

void data_update_adam(data_t *data, double rate)
{
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

    if (level > 0)
    {
        data->grad = buf;
        buf += data->size;
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
