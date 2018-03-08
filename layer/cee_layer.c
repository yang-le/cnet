#include "cee_layer.h"
#include "log.h"
#include <math.h>

static void cee_layer_prepare(layer_t *l)
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

	LOG("cee_layer: in %d\n", l->in.size);
}

static void cee_layer_forward(layer_t *l)
{
	int i = 0, b = 0;

	for (b = 0; b < l->n->batch; ++b)
	{
		l->out.val[b] = 0;
		for (i = 0; i < l->in.size; ++i)
		{
			// see https://www.zhihu.com/question/52242037
			if (l->in.val[b * l->in.size + i] < 1e-10)
				l->in.val[b * l->in.size + i] = 1e-10;

			l->out.val[b] += -l->extra.val[b * l->in.size + i] * log(l->in.val[b * l->in.size + i]);
		}
	}
}

static void cee_layer_backward(layer_t *l)
{
	int i = 0, b = 0;

	for (b = 0; b < l->n->batch; ++b)
		for (i = 0; i < l->in.size; ++i)
		{
			l->in.grad[b * l->in.size + i] = l->out.grad[b] * (-l->extra.val[b * l->in.size + i] / l->in.val[b * l->in.size + i]);
		}
}

static const layer_func_t cec_func = {
	cee_layer_prepare,
	cee_layer_forward,
	cee_layer_backward};

layer_t *cee_layer(int in)
{
	layer_t *l = layer(in, 1, 0, 0, in, &cec_func);

	return l;
}
