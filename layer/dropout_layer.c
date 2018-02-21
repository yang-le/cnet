#include "dropout_layer.h"
#include "log.h"
#include "random.h"
#include <math.h>

static void dropout_layer_prepare(layer_t *l)
{
	if (l->in.size != 0)
	{
		l->out.size = l->in.size;
	}

	if (l->out.size != 0)
	{
		l->in.size = l->out.size;
	}

	if (l->param.size != 1)
	{
		l->param.size = 1;
	}

	LOG("dropout_layer: in %d, out %d, param %d\n", l->in.size, l->out.size, l->param.size);
}

static void dropout_layer_forward(layer_t *l)
{
	int i = 0;

    // just use grad to store a random number
    uniform(l->in.grad, l->in.size, 0, 1);

	for (i = 0; i < l->in.size; ++i)
	{
        if (l->in.grad[i] < l->param.val[0])
            l->out.val[i] = 0;
        else
            l->out.val[i] = l->in.val[i] / (1 - l->param.val[0]);
	}
}

static void dropout_layer_backward(layer_t *l)
{
	int i = 0;
	for (i = 0; i < l->in.size; ++i)
	{
		if (l->in.grad[i] < l->param.val[0])
            l->in.grad[i] = 0;
        else
		    l->in.grad[i] += l->out.grad[i] / (1 - l->param.val[0]);
	}
}

static const layer_func_t dropout_func = {
	dropout_layer_prepare,
	dropout_layer_forward,
	dropout_layer_backward
};

layer_t* dropout_layer(int in, int out, int param)
{
	layer_t *l = layer(in, out, param, &dropout_func);

	return l;
}
