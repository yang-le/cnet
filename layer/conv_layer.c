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

	// I have to know ic, k, oh, ow

	//if (conv->oh == 0)
	//{
	//	conv->oh = (conv->ih + 2 * conv->p - conv->k) / conv->s + 1;
	//}

	//if (conv->ow == 0)
	//{
	//	conv->ow = (conv->iw + 2 * conv->p - conv->k) / conv->s + 1;
	//}

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
	l->param.size = conv->oc * (conv->ic * conv->k * conv->k + 1);

	conv->col.size = conv->ic * conv->k * conv->k * conv->oh * conv->ow;
	data_init(&conv->col, (data_val_t *)(conv + 1), 0, 1);

	LOG("conv_layer: %d x %d x %d => %d x %d x %d, kernel %d x %d + %d, padding %d, params %d\n",
		conv->ic, conv->iw, conv->ih, conv->oc, conv->ow, conv->oh, conv->k, conv->k, conv->s, conv->p, l->param.size);
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
		im2col(&l->in.val, b * l->in.size, conv->ic, conv->ih, conv->iw, conv->k, conv->s, conv->p, &conv->col.val, 0);
		gemm(0, 0, m, n, k, 1, &l->param.val, 0, k, &conv->col.val, 0, n, 0, &l->out.val, b * l->out.size, n);

		for (i = 0; i < m; ++i)
			for (j = 0; j < n; ++j)
			{
				l->out.val[b * l->out.size + i * n + j] += l->param.val[m * k + i];
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

		gemm(0, 1, m, n, k, 1, &l->out.grad, b * l->out.size, k, &conv->col.val, 0, k, 1, &l->param.grad, 0, n);

		m = conv->ic * conv->k * conv->k;
		n = conv->oh * conv->ow;
		k = conv->oc;

		gemm(1, 0, m, n, k, 1, &l->param.val, 0, m, &l->out.grad, b * l->out.size, n, 0, &conv->col.grad, 0, n);
		
		memset(l->in.grad + b * l->in.size, 0, l->in.size * sizeof(l->in.grad[0]));
		col2im(&conv->col.grad, 0, conv->ic, conv->ih, conv->iw, conv->k, conv->s, conv->p, &l->in.grad, b * l->in.size);

		for (i = 0; i < k; ++i)
			for (j = 0; j < n; ++j)
			{
				l->param.grad[k * m + i] += l->out.grad[b * l->out.size + i * n + j];
			}
	}
}

static const layer_func_t conv_func = {
	conv_layer_prepare,
	conv_layer_forward,
	conv_layer_backward};

layer_t *conv_layer(int ic, int iw, int ih, int oc, int ow, int oh, int k, int s, int p)
{
	conv_layer_t *conv = (conv_layer_t *)alloc(1, sizeof(conv_layer_t) +
													  ic * k * k * oh * ow * sizeof(data_val_t));

	conv->l.func = &conv_func;

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
