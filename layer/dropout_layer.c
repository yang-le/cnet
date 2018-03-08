#include "dropout_layer.h"
#include "log.h"
#include "random.h"
#include "common.h"
#include <math.h>

static void dropout_layer_prepare(layer_t *l)
{
	if (l->in.size != 0)
	{
		l->out.size = l->in.size;
		l->extra.size = l->in.size;
	}

	if (l->out.size != 0)
	{
		l->in.size = l->out.size;
		l->extra.size = l->out.size;
	}

	if (l->extra.size != 0)
	{
		l->in.size = l->extra.size;
		l->out.size = l->extra.size;
	}

	LOG("dropout_layer: in %d\n", l->in.size);
}

static void dropout_layer_forward(layer_t *l)
{
	int i = 0, b = 0;
	dropout_layer_t *drop = (dropout_layer_t *)l;

	if (l->n->train)
		for (b = 0; b < l->n->batch; ++b)
		{
			if (drop->prob > 0)
				uniform(l->extra.val, l->extra.size, 0, 1);

			for (i = 0; i < l->in.size; ++i)
			{
				if (l->extra.val[i] < drop->prob)
					l->out.val[b * l->in.size + i] = 0;
				else
					l->out.val[b * l->in.size + i] = l->in.val[b * l->in.size + i] / (1 - drop->prob);
			}
		}
	else
		for (b = 0; b < l->n->batch; ++b)
			for (i = 0; i < l->in.size; ++i)
				l->out.val[b * l->in.size + i] = l->in.val[b * l->in.size + i];
}

static void dropout_layer_backward(layer_t *l)
{
	int i = 0, b = 0;
	dropout_layer_t *drop = (dropout_layer_t *)l;

	for (b = 0; b < l->n->batch; ++b)
		for (i = 0; i < l->in.size; ++i)
		{
			if (l->extra.val[i] < drop->prob)
				l->in.grad[b * l->in.size + i] = 0;
			else
				l->in.grad[b * l->in.size + i] = l->out.grad[b * l->in.size + i] / (1 - drop->prob);
		}
}

static const layer_func_t dropout_func = {
	dropout_layer_prepare,
	dropout_layer_forward,
	dropout_layer_backward};

layer_t *dropout_layer(int n, float keeprob)
{
	dropout_layer_t *drop = (dropout_layer_t *)alloc(1, sizeof(dropout_layer_t));

	drop->prob = 1 - keeprob;

	drop->l.in.size = n;
	drop->l.out.size = n;
	drop->l.extra.size = n;
	drop->l.func = &dropout_func;

	return (layer_t *)drop;
}
