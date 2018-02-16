#include "softmax_layer.h"
#include "log.h"
#include <math.h>

void softmax_layer_prepare(layer_t *l)
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

	LOG("softmax_layer: in %d, out %d, param %d\n", l->in.size, l->out.size, l->param.size);
}

void softmax_layer_forward(layer_t *l)
{
	int i = 0;
	float sum = 0;

	for (i = 0; i < l->out.size; ++i)
	{
		l->out.val[i] = exp(l->in.val[i]);
		sum += l->out.val[i];
	}

	for (i = 0; i < l->out.size; ++i)
	{
		l->out.val[i] /= sum;
	}
}

void softmax_layer_backward(layer_t *l)
{
	int o = 0;
	for (o = 0; o < l->out.size; ++o)
	{
		int i = 0;
		for (i = 0; i < l->in.size; ++i)
		{
			l->in.grad[i] += l->out.grad[o] * l->out.val[o] * ((i == o) - l->out.val[i]);
		}
	}
}

static const layer_func_t softmax_func = {
	softmax_layer_prepare,
	softmax_layer_forward,
	softmax_layer_backward
};

layer_t* softmax_layer(int in, int out, int param)
{
	layer_t *l = layer(in, out, param, &softmax_func);

	return l;
}
