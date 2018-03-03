#include "mse_layer.h"
#include "log.h"

static void mse_layer_prepare(layer_t *l)
{
	l->out.size = 1;
	l->param.size = l->n->batch;

	LOG("mse_layer: in %d, out %d, param %d\n", l->in.size, l->out.size, l->param.size);
}

static void mse_layer_forward(layer_t *l)
{
	int i = 0, b = 0;

	for (b = 0; b < l->n->batch; ++b)
	{
		l->out.val[b] = 0;
		for (i = 0; i < l->in.size; ++i)
		{
			l->out.val[b] += (l->in.val[b * l->in.size + i] - l->param.val[b]) * (l->in.val[b * l->in.size + i] - l->param.val[b]) / 2;
		}
	}	
}

static void mse_layer_backward(layer_t *l)
{
	int i = 0, b = 0;

	for (b = 0; b < l->n->batch; ++b)
	for (i = 0; i < l->in.size; ++i)
	{
		l->in.grad[b * l->in.size + i] = l->out.grad[b] * (l->in.val[b * l->in.size + i] - l->param.val[b]);
	}
}

static const layer_func_t mse_func = {
	mse_layer_prepare,
	mse_layer_forward,
	mse_layer_backward
};

layer_t* mse_layer(int in, int out, int param)
{
	layer_t *l = layer(in, out, param, &mse_func);

	return l;
}
