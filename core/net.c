#include <stdlib.h>
#include <time.h>
#include <memory.h>
#include <math.h>

#include "net.h"
#include "log.h"
#include "random.h"
#include "common.h"

net_t *net_create(int size, int level, int batch)
{
	net_t *n = (net_t *)alloc(1, sizeof(net_t) + sizeof(layer_t *) * size);

	n->size = size;
	n->level = level;
	n->batch = batch;
	n->layer = (layer_t **)(n + 1);

	return n;
}

void net_add(net_t *n, struct layer *l)
{
	l->n = n;
	*n->layer++ = l;
}

void net_finish(net_t *n)
{
	int i = 0;
	int out_size = 0;
	int param_size = 0;
	int total_size = 0;
	data_val_t *data_buf = NULL;

	n->layer = (layer_t **)(n + 1);

	for (i = 0; i < n->size; ++i)
	{
		PREPARE(n->layer[i]);

		if ((i + 1 < n->size) && (0 == n->layer[i + 1]->in.size))
			n->layer[i + 1]->in.size = n->layer[i]->out.size;

		out_size += n->layer[i]->out.size;
		param_size += n->layer[i]->param.size;
	}

	total_size = (n->layer[0]->in.size * n->batch + param_size + out_size * n->batch) * (n->level + 1);
	data_buf = (data_val_t *)alloc(total_size, sizeof(data_val_t));
	if (data_buf == NULL)
	{
		LOG("fail to alloc %ld bytes memory, you can try with a smaller batch size.\n", total_size * sizeof(data_val_t));
		exit(-1);
	}

	for (i = 0; i < n->size; ++i)
	{
		data_buf += layer_data_init(n->layer[i], data_buf);
	}

	srand((unsigned int)time(NULL));
	for (i = 0; i < n->size; ++i)
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

	LOG("total: layers %d, params %d, heap size %ld\n", n->size, param_size, get_alloc_size());
}

void net_forward(net_t *n)
{
	int i = 0;
	for (i = 0; i < n->size; ++i)
	{
		FORWARD(n->layer[i]);
	}
}

void net_backward(net_t *n)
{
	int i = 0, b = 0;

	for (b = 0; b < n->batch; ++b)
	for (i = 0; i < LAST_LAYER(n)->out.size; ++i)
	{
		LAST_LAYER(n)->out.grad[b * LAST_LAYER(n)->out.size + i] = 1;
	}

	for (i = n->size - 1; i >= 0; --i)
	{
		BACKWARD(n->layer[i]);
	}
}

void net_update(net_t *n)
{
	int i = 0;

	if (n->level == TRAIN_ADAM)
		for (i = 0; i < n->size; ++i)
		{
			data_update_adam(&n->layer[i]->param);
		}
	else
		for (i = 0; i < n->size; ++i)
		{
			data_update(&n->layer[i]->param, n->rate);
		}
}

void net_destroy(net_t *n)
{
	int i = 0;

	free(n->layer[0]->in.val);

	for (i = 0; i < n->size; ++i)
	{
		free(n->layer[i]);
	}

	free(n);
}

void net_train(net_t *n, feed_func_t feed, float rate)
{
	int i = 0, b = 0;
	float loss = 0;

	n->rate = rate / n->batch;

	feed(n);

	net_forward(n);

	net_backward(n);

	for (b = 0; b < n->batch; ++b)
		loss += LAST_LAYER(n)->out.val[b];

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
	LOG("%f\n", loss / n->batch);
}

void net_param_save(net_t *n, const char *file)
{
	int i = 0;
	FILE *fp = fopen(file, "wb");

	for (i = 0; i < n->size; ++i)
	{
		data_save(&n->layer[i]->param, fp);
	}

	fclose(fp);
}

void net_param_load(net_t *n, const char *file)
{
	int i = 0;
	FILE *fp = fopen(file, "rb");

	if (fp == NULL)
	{
		LOG("param file %s not exist!\n", file);
		return;
	}

	for (i = 0; i < n->size; ++i)
	{
		data_load(fp, &n->layer[i]->param);
	}

	fclose(fp);
}
