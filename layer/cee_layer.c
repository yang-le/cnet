#include "cee_layer.h"
#include "log.h"
#include <math.h>

static void cee_layer_prepare(layer_t *l)
{
	if (l->in.size != 0)
	{
		l->param.size = l->in.size;
	}

	if (l->out.size != 1)
	{
		l->out.size = 1;
	}

	if (l->param.size != 0)
	{
		l->in.size = l->param.size;
	}

	LOG("cee_layer: in %d, out %d, param %d\n", l->in.size, l->out.size, l->param.size);
}

static void cee_layer_forward(layer_t *l)
{
	int i = 0;

	l->out.val[0] = 0;
	for (i = 0; i < l->in.size; ++i)
	{
		// see https://www.zhihu.com/question/52242037
		if (l->in.val[i] < 1e-10)
			l->in.val[i] = 1e-10;

		l->out.val[0] += -l->param.val[i] * log(l->in.val[i]);
	}
}

static void cee_layer_backward(layer_t *l)
{
	int i = 0;
	for (i = 0; i < l->in.size; ++i)
	{
		data_val_t grad = l->out.grad[0] * (-l->param.val[i] / l->in.val[i]);
		l->in.grad[i] += grad;
	}
}

static const layer_func_t cec_func = {
	cee_layer_prepare,
	cee_layer_forward,
	cee_layer_backward
};

layer_t* cee_layer(int in, int out, int param)
{
	layer_t *l = layer(in, out, param, &cec_func);

	return l;
}
