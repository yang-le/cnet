#include <math.h>
#include <stdlib.h>
#include <memory.h>

#include "conv_layer.h"
#include "log.h"
#include "gemm.h"
#include "im2col.h"
#include "common.h"

static void conv_layer_prepare(layer_t *l)
{
	conv_layer_t *conv = (conv_layer_t *)l;

	if (conv->s == 0)
	{
		conv->s = 1;
	}

	if (conv->ih == 0)
	{
		conv->ih = (conv->oh - 1) * conv->s + conv->k - 2 * conv->p;
	}

	if (conv->iw == 0)
	{
		conv->iw = (conv->ow - 1) * conv->s + conv->k - 2 * conv->p;
	}

	if (conv->p == 0)
	{
		conv->p = ((conv->ow - 1) * conv->s + conv->k - conv->iw) / 2;
	}

	l->in.size = conv->ic * conv->iw * conv->ih;
	l->out.size = conv->oc * conv->ow * conv->oh;
	l->weight.size = conv->oc * conv->ic * conv->k * conv->k;
	l->bias.size = conv->oc;
	l->extra.size = conv->ic * conv->k * conv->k * conv->oh * conv->ow;

	LOG("conv_layer: %d x %d x %d => %d x %d x %d, kernel %d x %d + %d, padding %d, params %d\n",
		conv->ic, conv->iw, conv->ih, conv->oc, conv->ow, conv->oh, conv->k, conv->k, conv->s, conv->p, l->weight.size + l->bias.size);
}

static void conv_layer_forward(layer_t *l)
{
	int i = 0, j = 0, b = 0;
	conv_layer_t *conv = (conv_layer_t *)l;

	int m = conv->oc;
	int k = conv->ic * conv->k * conv->k;
	int n = conv->oh * conv->ow;

	for (b = 0; b < l->n->batch; ++b)
	{
		im2col(&l->in.val, b * l->in.size, conv->ic, conv->ih, conv->iw, conv->k, conv->s, conv->p, &l->extra.val, 0);
		gemm(0, 0, m, n, k, 1, &l->weight.val, 0, k, &l->extra.val, 0, n, 0, &l->out.val, b * l->out.size, n);

		for (i = 0; i < m; ++i)
			for (j = 0; j < n; ++j)
			{
				l->out.val[b * l->out.size + i * n + j] += l->bias.val[i];
			}
	}
}

static void conv_layer_backward(layer_t *l)
{
	int i = 0, j = 0, b = 0;
	conv_layer_t *conv = (conv_layer_t *)l;

	for (b = 0; b < l->n->batch; ++b)
	{
		int m = conv->oc;
		int n = conv->ic * conv->k * conv->k;
		int k = conv->oh * conv->ow;

		gemm(0, 1, m, n, k, 1, &l->out.grad, b * l->out.size, k, &l->extra.val, 0, k, 1, &l->weight.grad, 0, n);

		m = conv->ic * conv->k * conv->k;
		n = conv->oh * conv->ow;
		k = conv->oc;

		gemm(1, 0, m, n, k, 1, &l->weight.val, 0, m, &l->out.grad, b * l->out.size, n, 0, &l->extra.grad, 0, n);

		memset(l->in.grad + b * l->in.size, 0, l->in.size * sizeof(l->in.grad[0]));
		col2im(&l->extra.grad, 0, conv->ic, conv->ih, conv->iw, conv->k, conv->s, conv->p, &l->in.grad, b * l->in.size);

		for (i = 0; i < k; ++i)
			for (j = 0; j < n; ++j)
			{
				l->bias.grad[i] += l->out.grad[b * l->out.size + i * n + j];
			}
	}
}

static const layer_func_t conv_func = {
	conv_layer_prepare,
	conv_layer_forward,
	conv_layer_backward};

layer_t *conv_layer(int ic, int iw, int ih, int oc, int ow, int oh, int k, int s, int p, int filler, float p0, float p1)
{
	conv_layer_t *conv = (conv_layer_t *)alloc(1, sizeof(conv_layer_t));

	conv->l.func = &conv_func;
	layer_set_filler(&conv->l.weight_filler, filler, p0, p1);

	conv->ic = ic;
	conv->iw = iw;
	conv->ih = ih;

	conv->oc = oc;
	conv->ow = ow;
	conv->oh = oh;

	conv->k = k;
	conv->s = s;
	conv->p = p;

	return (layer_t *)conv;
}
