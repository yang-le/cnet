#include "normalization_layer.h"
#include "common.h"
#include "log.h"
#include <math.h>

static void norm_helper(data_val_t *data, int size, data_val_t *mean, data_val_t *std_dev)
{
    const double EPSILON = 1e-8;
    
    data_val_t sum = 0;   
    for (int i = 0; i < size; ++i)
    {
        sum += data[i];
    }

    *mean = sum / size;

    for (int i = 0, sum = 0; i < size; ++i)
    {
        sum += (data[i] - *mean) * (data[i] - *mean);
    }

    *std_dev = sqrt(sum / size + EPSILON);
}

static void norm_layer_prepare(layer_t *l)
{
	if (l->in.size == 0)
	{
		l->in.size = l->out.size;
	}

	if (l->out.size == 0)
	{
		l->out.size = l->in.size;
	}

	LOG("normalization_layer: in %d\n", l->in.size);
}

static void bn_layer_prepare(layer_t *l)
{
    l->extra.size = 2 * l->n->batch;
    norm_layer_prepare(l);
}

static void bn_layer_forward(layer_t *l)
{

}

static void bn_layer_backward(layer_t *l)
{

}

static const layer_func_t bn_func = {
	bn_layer_prepare,
	bn_layer_forward,
	bn_layer_backward};

layer_t *bn_layer(int in)
{
    return layer(in, in, 0, 0, 0, &bn_func);
}
