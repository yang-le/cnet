#include <math.h>
#include <stdlib.h>
#include <memory.h>

#include "pooling_layer.h"
#include "log.h"
#include "im2col.h"

static void pooling_layer_prepare(layer_t *l)
{
	pooling_layer_t *pooling = (pooling_layer_t *)l;

	if (pooling->ic != pooling->oc)
	{
		pooling->oc = pooling->ic;
	}

	if (pooling->k == 0)
	{
		pooling->k = pooling->iw / pooling->ow;
	}

	if (pooling->s == 0)
	{
		pooling->s = pooling->k;
	}

	// I have to know c, k, oh, ow

	//if (pooling->oh == 0)
	//{
	//	pooling->oh = (pooling->ih + 2 * pooling->p - pooling->k) / pooling->s + 1;
	//}

	//if (pooling->ow == 0)
	//{
	//	pooling->ow = (pooling->iw + 2 * pooling->p - pooling->k) / pooling->s + 1;
	//}

	if (pooling->ih == 0)
	{
		pooling->ih = (pooling->oh - 1) * pooling->s + pooling->k - 2 * pooling->p;
	}

	if (pooling->iw == 0)
	{
		pooling->iw = (pooling->ow - 1) * pooling->s + pooling->k - 2 * pooling->p;
	}

	if (pooling->p == 0)
	{
		pooling->p = ((pooling->ow - 1) * pooling->s + pooling->k - pooling->iw) / 2;
	}

	l->in.size = pooling->ic * pooling->iw * pooling->ih;
	l->out.size = pooling->oc * pooling->ow * pooling->oh;
	l->param.size = 0;

	pooling->col.size = pooling->ic * pooling->k * pooling->k * pooling->oh * pooling->ow;
	pooling->col.val = (data_val_t *)(pooling + 1);
	pooling->col.grad = pooling->col.val;

	LOG("pooling_layer: %d x %d x %d => %d x %d x %d, kernel %d x %d + %d, padding %d, params %d\n",
		pooling->ic, pooling->iw, pooling->ih, pooling->oc, pooling->ow, pooling->oh, pooling->k, pooling->k, pooling->s, pooling->p, l->param.size);
}

static void max_pooling_layer_forward(layer_t *l)
{
	int i = 0, j = 0, k = 0;
	pooling_layer_t *pooling = (pooling_layer_t *)l;

	int channels = pooling->ic;
	int pool_size = pooling->k * pooling->k;
	int out_size = pooling->oh * pooling->ow;

	im2col(l->in.val, pooling->ic, pooling->ih, pooling->iw, pooling->k, pooling->s, pooling->p, pooling->col.val);
	
	for (k = 0; k < channels; ++k)
	{
		int offset = k * out_size;
		int pool_offset = pool_size * offset;

		for (j = 0; j < out_size; ++j)
		{
			l->out.val[offset + j] = pooling->col.val[pool_offset + j];
		}

		for (i = 1; i < pool_size; ++i)
		{
			int index = pool_offset  + i * out_size;
			for (j = 0; j < out_size; ++j)
			{
				if (pooling->col.val[index + j] > l->out.val[offset + j])
					l->out.val[offset + j] = pooling->col.val[index + j];
			}
		}
	}
}

static void max_pooling_layer_backward(layer_t *l)
{
	int i = 0, j = 0, k = 0;
	pooling_layer_t *pooling = (pooling_layer_t *)l;

	int channels = pooling->ic;
	int pool_size = pooling->k * pooling->k;
	int out_size = pooling->oh * pooling->ow;
	
	for (k = 0; k < channels; ++k)
	{
		int offset = k * out_size;
		for (i = 0; i < pool_size; ++i)
		{
			int index = (k * pool_size  + i) * out_size; 
			for (j = 0; j < out_size; ++j)
			{
				if (pooling->col.val[index + j] != l->out.val[offset + j])
					pooling->col.grad[index + j] = 0;
				else
					pooling->col.grad[index + j] = l->out.grad[offset + j];		
			}
		}
	}

	memset(l->in.grad, 0, l->in.size * sizeof(l->in.grad[0]));
	col2im(pooling->col.grad, pooling->ic, pooling->ih, pooling->iw, pooling->k, pooling->s, pooling->p, l->in.grad);
}

static void avg_pooling_layer_forward(layer_t *l)
{

}

static void avg_pooling_layer_backward(layer_t *l)
{

}

static const layer_func_t max_pooling_func = {
	pooling_layer_prepare,
	max_pooling_layer_forward,
	max_pooling_layer_backward
};

static const layer_func_t avg_pooling_func = {
	pooling_layer_prepare,
	avg_pooling_layer_forward,
	avg_pooling_layer_backward
};

static layer_t * pooling_layer(int c, int iw, int ih, int ow, int oh, int k, int s, int p, const layer_func_t *func)
{
	pooling_layer_t *pooling = (pooling_layer_t *)calloc(sizeof(pooling_layer_t) +
		c * k * k * oh * ow * sizeof(data_val_t), 1);

	pooling->l.func = func;

	pooling->ic = c;
	pooling->iw = iw;
	pooling->ih = ih;

	pooling->oc = c;
	pooling->ow = ow;
	pooling->oh = oh;

	pooling->k = k;
	pooling->s = s;
	pooling->p = p;

	return (layer_t*)pooling;
}

layer_t * max_pooling_layer(int c, int iw, int ih, int ow, int oh, int k, int s, int p)
{
	return pooling_layer(c, iw, ih, ow, oh, k, s, p, &max_pooling_func);
}

layer_t * avg_pooling_layer(int c, int iw, int ih, int ow, int oh, int k, int s, int p)
{
	return pooling_layer(c, iw, ih, ow, oh, k, s, p, &avg_pooling_func);
}
