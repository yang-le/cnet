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
		l->param.size = l->in.size;
	}

	if (l->out.size != 0)
	{
		l->in.size = l->out.size;
		l->param.size = l->out.size;
	}

	if (l->param.size != 0)
	{
		l->in.size = l->param.size;
		l->out.size = l->param.size;
	}

	LOG("dropout_layer: in %d, out %d, param %d\n", l->in.size, l->out.size, l->param.size);
}

static void dropout_layer_forward(layer_t *l)
{
	int i = 0;
	dropout_layer_t *drop = (dropout_layer_t*)l;

	if (drop->prob > 0)
		uniform(l->param.val, l->param.size, 0, 1);

	for (i = 0; i < l->in.size; ++i)
	{
		if (l->param.val[i] < drop->prob)
			l->out.val[i] = 0;
		else
			l->out.val[i] = l->in.val[i] / (1 - drop->prob);
	}
}

static void dropout_layer_backward(layer_t *l)
{
	int i = 0;
	dropout_layer_t *drop = (dropout_layer_t*)l;
	
	for (i = 0; i < l->in.size; ++i)
	{
		if (l->param.val[i] < drop->prob)
			l->in.grad[i] = 0;
		else
			l->in.grad[i] = l->out.grad[i] / (1 - drop->prob);
	}
}

static const layer_func_t dropout_func = {
	dropout_layer_prepare,
	dropout_layer_forward,
	dropout_layer_backward
};

layer_t* dropout_layer(int n, float droprob)
{
	dropout_layer_t *drop = (dropout_layer_t *)alloc(1, sizeof(dropout_layer_t));

	drop->prob = droprob;

	drop->l.in.size = n;
	drop->l.out.size = n;
	drop->l.param.size = n;
	drop->l.param.immutable = 1;
	drop->l.func = &dropout_func;

	return (layer_t*)drop;
}
