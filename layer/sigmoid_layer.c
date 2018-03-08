#include "sigmoid_layer.h"
#include "log.h"
#include <math.h>

static void sigmoid_layer_prepare(layer_t *l)
{
	if (l->in.size == 0)
	{
		l->in.size = l->out.size;
	}

	if (l->out.size == 0)
	{
		l->out.size = l->in.size;
	}

	LOG("sigmoid_layer: in %d\n", l->in.size);
}

static void sigmoid_layer_forward(layer_t *l)
{
	int i = 0, b = 0;

	for (b = 0; b < l->n->batch; ++b)
		for (i = 0; i < l->out.size; ++i)
		{
			l->out.val[b * l->out.size + i] = 1 / (1 + exp(-l->in.val[b * l->in.size + i]));
		}
}

static void sigmoid_layer_backward(layer_t *l)
{
	int i = 0, b = 0;

	for (b = 0; b < l->n->batch; ++b)
		for (i = 0; i < l->in.size; ++i)
		{
			l->in.grad[b * l->in.size + i] = l->out.grad[b * l->out.size + i] * l->out.val[b * l->out.size + i] * (1 - l->out.val[b * l->out.size + i]);
		}
}

static const layer_func_t sigmoid_func = {
	sigmoid_layer_prepare,
	sigmoid_layer_forward,
	sigmoid_layer_backward};

layer_t *sigmoid_layer(int in)
{
	layer_t *l = layer(in, in, 0, 0, 0, &sigmoid_func);

	return l;
}
