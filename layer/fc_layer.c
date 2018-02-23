#include "fc_layer.h"
#include "log.h"
#include "gemm.h"

static void fc_layer_prepare(layer_t *l)
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

static void fc_layer_forward(layer_t *l)
{
	int i = 0;

	int m = 1;
	int k = l->in.size;
	int n = l->out.size;
	float *a = l->in.val;
	float *b = l->param.val;
	float *c = l->out.val;

	gemm(0, 1, m, n, k, 1, a, k, b, k, 0, c, n);

	for (i = 0; i < l->out.size; ++i)
	{
		l->out.val[i] += l->param.val[l->out.size * l->in.size + i];
	}
}

static void fc_layer_backward(layer_t *l)
{
	int i = 0;

	int m = l->out.size;
	int k = 1;
	int n = l->in.size;
	float *a = l->out.grad;
	float *b = l->in.val;
	float *c = l->param.grad;

	gemm(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

	m = 1;
	k = l->out.size;
	n = l->in.size;
	a = l->out.grad;
	b = l->param.val;
	c = l->in.grad;

	gemm(0, 0, m, n, k, 1, a, k, b, n, 0, c, n);

	for (i = 0; i < l->out.size; ++i)
	{
		l->param.grad[l->out.size * l->in.size + i] += l->out.grad[i];
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
