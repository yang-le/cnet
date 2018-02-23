#include "mse_layer.h"
#include "log.h"

static void mse_layer_prepare(layer_t *l)
{
	if (l->in.size != 0)
	{
		l->param.size = l->in.size;
	}

	if (l->out.size != 1)
	{
		l->out.size = 1;
	}

	if (l->param.size != 0)
	{
		l->in.size = l->param.size;
	}

	LOG("mse_layer: in %d, out %d, param %d\n", l->in.size, l->out.size, l->param.size);
}

static void mse_layer_forward(layer_t *l)
{
	int i = 0;

	l->out.val[0] = 0;
	for (i = 0; i < l->in.size; ++i)
	{
		l->out.val[0] += (l->in.val[i] - l->param.val[i]) * (l->in.val[i] - l->param.val[i]) / 2;
	}	
}

static void mse_layer_backward(layer_t *l)
{
	int i = 0;
	for (i = 0; i < l->in.size; ++i)
	{
		l->in.grad[i] = l->out.grad[i] * (l->in.val[i] - l->param.val[i]);
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
