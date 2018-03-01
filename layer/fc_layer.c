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

	gemm(0, 1, m, n, k, 1, &l->in.val, k, &l->param.val, k, 0, &l->out.val, n);

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

	gemm(1, 0, m, n, k, 1, &l->out.grad, m, &l->in.val, n, 1, &l->param.grad, n);

	m = 1;
	k = l->out.size;
	n = l->in.size;

	gemm(0, 0, m, n, k, 1, &l->out.grad, k, &l->param.val, n, 0, &l->in.grad, n);

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
