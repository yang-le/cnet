#include "layer.h"
#include <stdlib.h>

layer_t* layer(int in, int out, int param, const layer_func_t *func)
{
	layer_t *l = (layer_t *)calloc(1, sizeof(layer_t));

	l->in.size = abs(in);
	l->in.mutable = (in >= 0);
	l->out.size = abs(out);
	l->out.mutable = (out >= 0);
	l->param.size = param;
	l->param.mutable = (param >= 0);
	l->func = func;

	return l;
}
