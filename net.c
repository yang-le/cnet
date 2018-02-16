#include <stdlib.h>

#include "net.h"
#include "layer.h"
#include "log.h"
#include <memory.h>
#include <math.h>

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
	LOG("total: layers %d, params %d, heap size %d\n", n->size, param_size, total_size * 2 * sizeof(data_val_t));

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

	for(i = 0; i < n->size; ++i)
	{
		int j = 0;

		//LOG("layer %d param ", i);

		if (n->layer[i]->param.mutable)
		for (j = 0; j < n->layer[i]->param.size; ++j)
		{
			n->layer[i]->param.val[j] = 1.0 * (rand() - RAND_MAX / 2) / RAND_MAX;
			//LOG("%f ", n->layer[i]->param.val[j]);
		}

		//LOG("\n");
	}
}

void net_forward(net_t *n)
{
	int i = 0;
	for(i = 0; i < n->size; ++i)
	{
		int j = 0;

		FORWARD(n->layer[i]);

		for (j = 0; j < n->layer[i]->in.size; ++j)
		{
			n->layer[i]->in.grad[j] = 0;
		}

		for (j = 0; j < n->layer[i]->param.size; ++j)
		{
			n->layer[i]->param.grad[j] = 0;
		}
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
		int j = 0;
		
		BACKWARD(n->layer[i]);
	}
	
	for(i = 0; i < n->size; ++i)
	{
		int j = 0;

		if (n->layer[i]->param.mutable)
		for (j = 0; j < n->layer[i]->param.size; ++j)
			n->layer[i]->param.val[j] -= n->rate * n->layer[i]->param.grad[j];

		if (n->layer[i]->in.mutable)
		for (j = 0; j < n->layer[i]->in.size; ++j)
			n->layer[i]->in.val[j] -= n->rate * n->layer[i]->in.grad[j];
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

	n->rate = rate;

	for (i = 0; i < round; ++i)
	{
		feed(n);

		net_forward(n);

		net_backward(n);

		loss +=  LAST_LAYER(n)->out.val[0];
	}

	//LOG("static:\n");
	for(i = 0; i < n->size; ++i)
	{
		int j = 0;

		if (n->layer[i]->param.mutable)
		{
			LOG("layer %d: ", i);
			for (j = 0; j < n->layer[i]->param.size; ++j)
			{
				LOG("w[%d] = %f ", j, n->layer[i]->param.val[j]);
			}
			LOG("\n");
		}
	}
	LOG("avg loss = %f\n",  loss / round);
}
