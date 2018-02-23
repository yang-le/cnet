#include "data.h"

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

void data_load(char *file, data_t *data)
{

}

void data_save(data_t *data, char *file)
{

}
