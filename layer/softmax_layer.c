#include "softmax_layer.h"
#include "log.h"
#include <math.h>

static void softmax_layer_prepare(layer_t *l)
{
	if (l->in.size == 0)
	{
		l->in.size = l->out.size;
	}

	if (l->out.size == 0)
	{
		l->out.size = l->in.size;
	}

	LOG("softmax_layer: in %d\n", l->in.size);
}

static void softmax_layer_forward(layer_t *l)
{
	int i = 0, b = 0;

	for (b = 0; b < l->n->batch; ++b)
	{
		float sum = 0;
		float max = l->in.val[b * l->in.size];

		// see http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability
		for (i = 1; i < l->in.size; ++i)
		{
			if (l->in.val[b * l->in.size + i] > max)
				max = l->in.val[b * l->in.size + i];
		}

		for (i = 0; i < l->out.size; ++i)
		{
			l->out.val[b * l->out.size + i] = exp(l->in.val[b * l->in.size + i] - max);
			sum += l->out.val[b * l->in.size + i];
		}

		for (i = 0; i < l->out.size; ++i)
		{
			l->out.val[b * l->out.size + i] /= sum;
		}
	}
}

static void softmax_layer_backward(layer_t *l)
{
	int i = 0, b = 0;

	for (b = 0; b < l->n->batch; ++b)
		for (i = 0; i < l->in.size; ++i)
		{
			int o = 0;

			l->in.grad[b * l->in.size + i] = 0;
			for (o = 0; o < l->out.size; ++o)
			{
				l->in.grad[b * l->in.size + i] += l->out.grad[b * l->out.size + o] * l->out.val[b * l->out.size + o] * ((i == o) - l->out.val[b * l->out.size + i]);
			}
		}
}

static const layer_func_t softmax_func = {
	softmax_layer_prepare,
	softmax_layer_forward,
	softmax_layer_backward};

layer_t *softmax_layer(int in)
{
	layer_t *l = layer(in, in, 0, 0, 0, &softmax_func);

	return l;
}
