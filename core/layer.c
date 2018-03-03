#include "layer.h"
#include "common.h"
#include <stdlib.h>

layer_t* layer(int in, int out, int param, const layer_func_t *func)
{
	layer_t *l = (layer_t *)alloc(1, sizeof(layer_t));

	l->in.size = abs(in);
	l->in.immutable = (in < 0);
	l->out.size = abs(out);
	l->out.immutable = (out < 0);
	l->param.size = param;
	l->param.immutable = (param < 0);
	l->func = func;

	return l;
}

size_t layer_data_init(layer_t *l, data_val_t *buf)
{
	data_val_t *start = buf;

	buf += data_init(&l->in, buf, l->n->level, l->n->batch);
	buf +=  data_init(&l->param, buf, l->n->level, 1);
/*  buf += */data_init(&l->out, buf, l->n->level, l->n->batch);

	return buf - start;
}
