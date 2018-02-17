#include "sigmoid_layer.h"
#include "log.h"
#include <math.h>

void sigmoid_layer_prepare(layer_t *l)
{
	if (l->in.size == 0)
	{
		l->in.size = l->out.size;
	}

	if (l->out.size == 0)
	{
		l->out.size = l->in.size;
	}

	if (l->param.size != 0)
	{
		l->param.size = 0;
	}

	LOG("sigmoid_layer: in %d, out %d, param %d\n", l->in.size, l->out.size, l->param.size);
}

void sigmoid_layer_forward(layer_t *l)
{
	int i = 0;
	for (i = 0; i < l->out.size; ++i)
	{
		l->out.val[i] = 1 / (1 + exp(-l->in.val[i]));
	}
}

void sigmoid_layer_backward(layer_t *l)
{
	int i = 0;
	for (i = 0; i < l->in.size; ++i)
	{
		l->in.grad[i] += l->out.grad[i] * l->out.val[i] * (1 - l->out.val[i]);
	}
}

static const layer_func_t sigmoid_func = {
	sigmoid_layer_prepare,
	sigmoid_layer_forward,
	sigmoid_layer_backward
};

layer_t* sigmoid_layer(int in, int out, int param)
{
	layer_t *l = layer(in, out, param, &sigmoid_func);

	return l;
}
