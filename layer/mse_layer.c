#include "mse_layer.h"
#include "log.h"

static void mse_layer_prepare(layer_t *l)
{
	if (l->in.size != 0)
	{
		l->extra.size = l->in.size;
	}

	if (l->out.size != 1)
	{
		l->out.size = 1;
	}

	if (l->extra.size != 0)
	{
		l->in.size = l->extra.size;
	}

	l->extra.size *= l->n->batch;

	LOG("mse_layer: in %d\n", l->in.size);
}

static void mse_layer_forward(layer_t *l)
{
	int i = 0, b = 0;

	for (b = 0; b < l->n->batch; ++b)
	{
		l->out.val[b] = 0;
		for (i = 0; i < l->in.size; ++i)
		{
			l->out.val[b] += (l->in.val[b * l->in.size + i] - l->extra.val[b * l->in.size + i]) * (l->in.val[b * l->in.size + i] - l->extra.val[b * l->in.size + i]) / 2;
		}
	}
}

static void mse_layer_backward(layer_t *l)
{
	int i = 0, b = 0;

	for (b = 0; b < l->n->batch; ++b)
		for (i = 0; i < l->in.size; ++i)
		{
			l->in.grad[b * l->in.size + i] = l->out.grad[b] * (l->in.val[b * l->in.size + i] - l->extra.val[b * l->in.size + i]);
		}
}

static const layer_func_t mse_func = {
	mse_layer_prepare,
	mse_layer_forward,
	mse_layer_backward};

layer_t *mse_layer(int in)
{
	layer_t *l = layer(in, 1, 0, 0, in, &mse_func);

	return l;
}
