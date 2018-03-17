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

static void layer_fill_data(data_filler_t *filler, data_t *data, int insize, int outsize)
{
	switch (filler->method)
	{
	case FILLER_CONST:
		for (int i = 0; i < data->size; ++i)
			data->val[i] = filler->param[0];
	case FILLER_GAUSS:
		normal(data->val, data->size, filler->param[0], filler->param[1]);
		break;
	case FILLER_UNIFORM:
		uniform(data->val, data->size, filler->param[0], filler->param[1]);
		break;
	case FILLER_XAVIER:
		uniform(data->val, data->size,
				-sqrt(3 / (filler->param[0] * insize + (1 - filler->param[0]) * outsize)),
				sqrt(3 / (filler->param[0] * insize + (1 - filler->param[0]) * outsize)));
		break;
	case FILLER_MSRA:
		normal(data->val, data->size, 0,
			   sqrt(2 / (filler->param[0] * insize + (1 - filler->param[0]) * outsize)));
		break;
	default:
		break;
	}
}

size_t layer_data_init(layer_t *l, data_val_t *buf, int level)
{
	data_val_t *start = buf;

	buf += data_init(&l->in, buf, level > 0, l->n->batch);
	buf += data_init(&l->weight, buf, level, 1);
	buf += data_init(&l->bias, buf, level, 1);
	buf += data_init(&l->extra, buf, 0, 1);
	/*  buf += */ data_init(&l->out, buf, level, l->n->batch);

	layer_fill_data(&l->weight_filler, &l->weight, l->in.size, l->out.size);
	layer_fill_data(&l->bias_filler, &l->bias, l->in.size, l->out.size);

	return buf - start;
}

void layer_set_filler(data_filler_t *filler, int method, data_val_t p0, data_val_t p1)
{
	filler->method = method;
	filler->param[0] = p0;
	filler->param[1] = p1;
}
