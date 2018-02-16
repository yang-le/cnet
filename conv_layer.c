#include "conv_layer.h"
#include "log.h"
#include <math.h>
#include <stdlib.h>

static void conv_layer_prepare(layer_t *l)
{
	conv_layer_t *conv = (conv_layer_t *)l;

	if (conv->s == 0)
	{
		conv->s = 1;
	}

	if (conv->oh == 0)
	{
		conv->oh = (conv->ih + 2 * conv->p - conv->k) / conv->s + 1;
	}

	if (conv->ow == 0)
	{
		conv->ow = (conv->iw + 2 * conv->p - conv->k) / conv->s + 1;
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
	l->param.size = conv->oc * (conv->ic * conv->k * conv->k + 1);

	LOG("conv_layer: %d x %d x %d => %d x %d x %d, kernel %d x %d + %d, padding %d, params %d\n",
		conv->ic, conv->iw, conv->ih, conv->oc, conv->ow, conv->oh, conv->k, conv->k, conv->s, conv->p, l->param.size);
}

static void conv_layer_forward(layer_t *l)
{

}

static void conv_layer_backward(layer_t *l)
{

}

static const layer_func_t conv_func = {
	conv_layer_prepare,
	conv_layer_forward,
	conv_layer_backward
};

layer_t * conv_layer(int ic, int iw, int ih, int oc, int ow, int oh, int k, int s, int p)
{
	conv_layer_t *conv = (conv_layer_t *)calloc(sizeof(conv_layer_t), 1);

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

	return (layer_t*)conv;
}
