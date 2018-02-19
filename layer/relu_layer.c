#include "relu_layer.h"
#include "log.h"

static void relu_layer_prepare(layer_t *l)
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

	LOG("relu_layer: in %d, out %d, param %d\n", l->in.size, l->out.size, l->param.size);
}

static void relu_layer_forward(layer_t *l)
{
	int i = 0;
	for (i = 0; i < l->out.size; ++i)
	{
		l->out.val[i] = 0;
		if (l->in.val[i] > 0)
			l->out.val[i] = l->in.val[i];
	}
}

static void relu_layer_backward(layer_t *l)
{
	int i = 0;
	for (i = 0; i < l->in.size; ++i)
	{
		l->in.grad[i] = l->out.grad[i] * (l->in.val[i] > 0);
	}
}

static const layer_func_t relu_func = {
	relu_layer_prepare,
	relu_layer_forward,
	relu_layer_backward
};

layer_t* relu_layer(int in, int out, int param)
{
	layer_t *l = layer(in, out, param, &relu_func);

	return l;
}
