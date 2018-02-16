#include "fc_layer.h"
#include "log.h"

void fc_layer_prepare(layer_t *l)
{
	if (l->in.size == 0)
	{
		l->in.size = l->param.size / l->out.size - 1;
	}

	if (l->out.size == 0)
	{
		l->out.size = l->param.size / (l->in.size + 1);
	}

	if (l->param.size == 0)
	{
		l->param.size = (l->in.size + 1) * l->out.size;
	}

	LOG("fc_layer: in %d, out %d, param %d\n", l->in.size, l->out.size, l->param.size);
}

void fc_layer_forward(layer_t *l)
{
	int o = 0;
	for (o = 0; o < l->out.size; ++o)
	{
		int i = 0;
		l->out.val[o] = 0;
		for (i = 0; i < l->in.size; ++i)
		{
			l->out.val[o] += l->param.val[o * (l->in.size + 1) + i] * l->in.val[i];
		}
		l->out.val[o] += l->param.val[o * (l->in.size + 1) + l->in.size];
	}
}

void fc_layer_backward(layer_t *l)
{
	int o = 0;
	for (o = 0; o < l->out.size; ++o)
	{
		int i = 0;
		for (i = 0; i < l->in.size; ++i)
		{
			l->in.grad[i] += l->out.grad[o] * l->param.val[o * (l->in.size + 1) + i];
			l->param.grad[o * (l->in.size + 1) + i] += l->out.grad[o] * l->in.val[i];
		}
		l->param.grad[o * (l->in.size + 1) + l->in.size] += l->out.grad[o];
	}
}

static const layer_func_t fc_func = {
	fc_layer_prepare,
	fc_layer_forward,
	fc_layer_backward
};

layer_t* fc_layer(int in, int out, int param)
{
	layer_t *l = layer(in, out, param, &fc_func);

	return l;
}
