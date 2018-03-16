#include "fc_layer.h"
#include "log.h"
#include "gemm.h"

static void fc_layer_prepare(layer_t *l)
{
	if (l->in.size == 0)
	{
		l->in.size = l->weight.size / l->out.size;
	}

	if (l->out.size == 0)
	{
		l->out.size = l->bias.size;
	}

	if (l->weight.size == 0)
	{
		l->weight.size = l->in.size * l->out.size;
	}

	if (l->bias.size == 0)
	{
		l->bias.size = l->out.size;
	}

	LOG("fc_layer: in %d, out %d, param %d\n", l->in.size, l->out.size, l->weight.size + l->bias.size);
}

static void fc_layer_forward(layer_t *l)
{
	int i = 0;
	int b = 0;

	int m = l->n->batch;
	int k = l->in.size;
	int n = l->out.size;

	gemm(0, 1, m, n, k, 1, &l->in.val, 0, k, &l->weight.val, 0, k, 0, &l->out.val, 0, n);

	for (b = 0; b < l->n->batch; ++b)
		for (i = 0; i < l->out.size; ++i)
		{
			l->out.val[b * l->out.size + i] += l->bias.val[i];
		}
}

static void fc_layer_backward(layer_t *l)
{
	int i = 0;
	int b = 0;

	int m = l->out.size;
	int k = l->n->batch;
	int n = l->in.size;

	gemm(1, 0, m, n, k, 1, &l->out.grad, 0, m, &l->in.val, 0, n, 1, &l->weight.grad, 0, n);

	m = l->n->batch;
	k = l->out.size;
	n = l->in.size;

	gemm(0, 0, m, n, k, 1, &l->out.grad, 0, k, &l->weight.val, 0, n, 0, &l->in.grad, 0, n);

	for (b = 0; b < l->n->batch; ++b)
		for (i = 0; i < l->out.size; ++i)
		{
			l->bias.grad[i] += l->out.grad[b * l->out.size + i];
		}
}

static const layer_func_t fc_func = {
	fc_layer_prepare,
	fc_layer_forward,
	fc_layer_backward};

layer_t *fc_layer(int in, int out, int filler, float p0, float p1)
{
	layer_t *l = layer(in, out, in * out, out, 0, &fc_func);
	layer_set_filler(&l->weight_filler, filler, p0, p1);
	return l;
}
