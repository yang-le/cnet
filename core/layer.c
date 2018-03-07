#include "layer.h"
#include "common.h"
#include "random.h"
#include <stdlib.h>

layer_t *layer(int in, int out, int weight, int bias, int extra, const layer_func_t *func)
{
	layer_t *l = (layer_t *)alloc(1, sizeof(layer_t));

	l->in.size = in;
	l->out.size = out;
	l->weight.size = weight;
	l->bias.size = bias;
	l->extra.size = extra;
	l->func = func;

	return l;
}

size_t layer_data_init(layer_t *l, data_val_t *buf, int level)
{
	data_val_t *start = buf;

	buf += data_init(&l->in, buf, level, l->n->batch);
	buf += data_init(&l->weight, buf, level, 1);
	buf += data_init(&l->bias, buf, level, 1);
	buf += data_init(&l->extra, buf, level, 1);
	/*  buf += */ data_init(&l->out, buf, level, l->n->batch);

	switch (l->weight_filler.method)
	{
	case FILLER_CONST:
		for (int i = 0; i < l->weight.size; ++i)
			l->weight.val[i] = l->weight_filler.param[0];
	case FILLER_GAUSS:
		normal(l->weight.val, l->weight.size, l->weight_filler.param[0], l->weight_filler.param[1]);
		break;
	case FILLER_UNIFORM:
		uniform(l->weight.val, l->weight.size, l->weight_filler.param[0], l->weight_filler.param[1]);
		break;
	case FILLER_XAVIER:
		uniform(l->weight.val, l->weight.size,
				-sqrt(3 / (l->weight_filler.param[0] * l->in.size + (1 - l->weight_filler.param[0]) * l->out.size)),
				sqrt(3 / (l->weight_filler.param[0] * l->in.size + (1 - l->weight_filler.param[0]) * l->out.size)));
		break;
	case FILLER_MSRA:
		normal(l->weight.val, l->weight.size, 0,
			   sqrt(2 / (l->weight_filler.param[0] * l->in.size + (1 - l->weight_filler.param[0]) * l->out.size)));
		break;
	default:
		break;
	}

	switch (l->bias_filler.method)
	{
	case FILLER_CONST:
		for (int i = 0; i < l->bias.size; ++i)
			l->bias.val[i] = l->bias_filler.param[0];
	case FILLER_GAUSS:
		normal(l->bias.val, l->bias.size, l->bias_filler.param[0], l->bias_filler.param[1]);
		break;
	case FILLER_UNIFORM:
		uniform(l->bias.val, l->bias.size, l->bias_filler.param[0], l->bias_filler.param[1]);
		break;
	case FILLER_XAVIER:
		uniform(l->bias.val, l->bias.size,
				-sqrt(3 / (l->bias_filler.param[0] * l->in.size + (1 - l->bias_filler.param[0]) * l->out.size)),
				sqrt(3 / (l->bias_filler.param[0] * l->in.size + (1 - l->bias_filler.param[0]) * l->out.size)));
		break;
	case FILLER_MSRA:
		normal(l->bias.val, l->bias.size, 0,
			   sqrt(2 / (l->bias_filler.param[0] * l->in.size + (1 - l->bias_filler.param[0]) * l->out.size)));
		break;
	default:
		break;
	}

	return buf - start;
}

void layer_set_weight_filler(layer_t *l, int method, data_val_t p0, data_val_t p1)
{
	l->weight_filler.method = method;
	l->weight_filler.param[0] = p0;
	l->weight_filler.param[1] = p1;
}

void layer_set_bias_filler(layer_t *l, int method, data_val_t p0, data_val_t p1)
{
	l->bias_filler.method = method;
	l->bias_filler.param[0] = p0;
	l->bias_filler.param[1] = p1;
}
