#include "scale_layer.h"
#include "log.h"

static void scale_layer_prepare(layer_t *l)
{
	if (l->in.size == 0)
	{
		l->in.size = l->out.size;
	}

	if (l->out.size == 0)
	{
		l->out.size = l->in.size;
	}

	LOG("scale_layer: in %d\n", l->in.size);
}

static void scale_layer_forward(layer_t *l)
{
	int i = 0, b = 0;

	for (b = 0; b < l->n->batch; ++b)
		for (i = 0; i < l->out.size; ++i)
		{
			l->out.val[b * l->out.size + i] = l->weight.val[0] * l->in.val[b * l->in.size + i] + l->bias.val[0];
		}
}

static void scale_layer_backward(layer_t *l)
{
	int i = 0, b = 0;

	for (b = 0; b < l->n->batch; ++b)
		for (i = 0; i < l->in.size; ++i)
		{
			l->in.grad[b * l->in.size + i] = l->weight.val[0] * l->out.grad[b * l->out.size + i];
            l->weight.grad[0] += l->in.val[b * l->in.size + i] * l->out.grad[b * l->out.size + i];
            l->bias.grad[0] += l->out.grad[b * l->out.size + i];
		}
}

static const layer_func_t scale_func = {
	scale_layer_prepare,
	scale_layer_forward,
	scale_layer_backward};

layer_t *scale_layer(int in, int filler, float p0, float p1)
{
	layer_t *l = layer(in, in, 1, 1, 0, &scale_func);
    layer_set_weight_filler(l, filler, p0, p1);
	return l;
}
