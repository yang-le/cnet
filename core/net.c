#include <stdlib.h>
#include <time.h>
#include <memory.h>
#include <math.h>

#include "net.h"
#include "layer.h"
#include "log.h"
#include "random.h"

net_t* net_create(size_t size)
{
	net_t *n = (net_t *)calloc(1, sizeof(net_t) + sizeof(layer_t*) * size);

	n->size = size;
	n->layer = (layer_t **)(n + 1);

	return n;
}

void net_add(net_t *n, layer_t *l)
{
	*n->layer++ = l;
}

void net_finish(net_t *n)
{
	int i = 0;
	int out_size = 0;
	int param_size = 0;
	int total_size = 0;
	data_val_t *temp = NULL;

	n->layer = (layer_t **)(n + 1);

	for(i = 0; i < n->size; ++i)
	{
		PREPARE(n->layer[i]);

		if ((i + 1 < n->size) && (0 == n->layer[i + 1]->in.size))
			n->layer[i + 1]->in.size = n->layer[i]->out.size;

		out_size += n->layer[i]->out.size;
		param_size += n->layer[i]->param.size;
	}
	
	total_size = n->layer[0]->in.size + param_size + out_size;
	LOG("total: layers %d, params %d, heap size %ld\n", n->size, param_size, total_size * 2 * sizeof(data_val_t));

	temp = (data_val_t*)calloc(total_size, 2 * sizeof(data_val_t));

	for(i = 0; i < n->size; ++i)
	{
		n->layer[i]->in.val = temp;
		temp += n->layer[i]->in.size;
		n->layer[i]->in.grad = temp;
		temp += n->layer[i]->in.size;

		n->layer[i]->param.val = temp;
		temp += n->layer[i]->param.size;
		n->layer[i]->param.grad = temp;
		temp += n->layer[i]->param.size;

		n->layer[i]->out.val = temp;
		n->layer[i]->out.grad = temp + n->layer[i]->out.size;
	}

	srand((unsigned int)time(NULL));
	for(i = 0; i < n->size; ++i)
	{
		int j = 0;

		//LOG("layer %d param ", i);

		if (!n->layer[i]->param.immutable)
		{
			for (j = 0; j < n->layer[i]->param.size - n->layer[i]->out.size; ++j)
			{
				truncated_normal(&n->layer[i]->param.val[j], 0, 0.1);
				//LOG("%f ", n->layer[i]->param.val[j]);
			}

			// init bias
			for (; j < n->layer[i]->param.size; ++j)
			{
				n->layer[i]->param.val[j] = 0.1;
				//LOG("%f ", n->layer[i]->param.val[j]);
			}
		}

		//LOG("\n");
	}
}

void net_forward(net_t *n)
{
	int i = 0;
	for(i = 0; i < n->size; ++i)
	{
		FORWARD(n->layer[i]);
	}
}

void net_backward(net_t *n)
{
	int i = 0;

	for (i = 0; i < LAST_LAYER(n)->out.size; ++i)
	{
		LAST_LAYER(n)->out.grad[i] = 1;
	}

	for(i = n->size - 1; i >= 0; --i)
	{
		BACKWARD(n->layer[i]);
	}
}

void net_update(net_t *n)
{
	int i = 0;

	for(i = 0; i < n->size; ++i)
	{
		data_update(&n->layer[i]->param, n->rate);
	}
}

void net_destroy(net_t *n)
{
	int i = 0;

	free(n->layer[0]->in.val);

	for(i = 0; i < n->size; ++i)
	{
		free(n->layer[i]);
	}

	free(n);
}

void net_train(net_t *n, feed_func_t feed, float rate, int round)
{
	int i = 0;
	float loss = 0;

	n->rate = rate / round;

	for (i = 0; i < round; ++i)
	{
		feed(n);

		net_forward(n);

		net_backward(n);

		loss +=  LAST_LAYER(n)->out.val[0];
	}

	net_update(n);

	//LOG("static:\n");
#if 0
	for(i = 0; i < n->size; ++i)
	{
		int j = 0;

		if (!n->layer[i]->param.immutable)
		{
			LOG("layer %d: ", i);
			for (j = 0; j < n->layer[i]->param.size; ++j)
			{
				LOG("w[%d] = %f ", j, n->layer[i]->param.val[j]);
			}
			LOG("\n");
		}
	}
#endif
	LOG("%f\n",  loss / round);
}
