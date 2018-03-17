#include <stdlib.h>
#include <time.h>
#include <memory.h>
#include <math.h>

#include "net.h"
#include "log.h"
#include "random.h"
#include "common.h"
#include "branch_layer.h"
#include "rnn_layer.h"

net_t *net_create(int size, int method, int batch)
{
	net_t *n = (net_t *)alloc(1, sizeof(net_t) + sizeof(layer_t *) * size);

	n->size = size;
	n->method = method;
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
	int extra_size = 0;
	int total_size = 0;
	int level[TRAIN_MAX] = {0, 1, 2, 2, 2, 2, 3};
	data_val_t *data_buf = NULL;

	n->layer = (layer_t **)(n + 1);

	for (i = 0; i < n->size; ++i)
	{
		PREPARE(n->layer[i]);

		if ((i + 1 < n->size) && (0 == n->layer[i + 1]->in.size))
			n->layer[i + 1]->in.size = n->layer[i]->out.size;

		out_size += n->layer[i]->out.size;
		param_size += n->layer[i]->weight.size + n->layer[i]->bias.size;
		extra_size += n->layer[i]->extra.size;
	}

	total_size = (n->layer[0]->in.size + out_size) * ((n->method != TRAIN_FORWARD) + 1) * n->batch + param_size * (level[n->method] + 1) + extra_size;
	data_buf = (data_val_t *)alloc(total_size, sizeof(data_val_t));
	if (data_buf == NULL)
	{
		LOG("fail to alloc %ld bytes memory, you can try with a smaller batch size.\n", total_size * sizeof(data_val_t));
		exit(-1);
	}

	srand((unsigned int)time(NULL));
	for (i = 0; i < n->size; ++i)
	{
		data_buf += layer_data_init(n->layer[i], data_buf, level[n->method]);
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
	int i = 0;

	if (n->method == TRAIN_NESTEROV)
		for (i = 0; i < n->size; ++i)
		{
			data_update_nesterov(&n->layer[i]->weight);
			data_update_nesterov(&n->layer[i]->bias);
		}

	for (i = n->size - 1; i >= 0; --i)
	{
		BACKWARD(n->layer[i]);
	}
}

void net_update(net_t *n)
{
	int i = 0;
	void (*update[TRAIN_MAX])(data_t *, data_val_t) = {
		NULL,
		data_update_sgd,
		data_update_momentum,
		data_update_momentum,
		data_update_adagrad,
		data_update_adadelta,
		NULL};

	if (update[n->method])
		for (i = 0; i < n->size; ++i)
		{
			if (is_branch_layer(n->layer[i]))
			{
				branch_layer_t *me = (branch_layer_t *)n->layer[i];
				for (int j = 0; j < me->num; ++j)
				{
					me->branch[j].n->rate = n->rate;
					net_update(me->branch[j].n);
				}
			}
			else if (is_rnn_layer(n->layer[i]))
			{
				rnn_layer_t *rnn = (rnn_layer_t *)n->layer[i];
				update[n->method](&n->layer[i]->weight, n->rate / rnn->len);
				update[n->method](&n->layer[i]->bias, n->rate / rnn->len);
			}
			else
			{
				update[n->method](&n->layer[i]->weight, n->rate);
				update[n->method](&n->layer[i]->bias, n->rate);
			}
		}
	else if (n->method == TRAIN_ADAM)
		for (i = 0; i < n->size; ++i)
		{
			if (is_branch_layer(n->layer[i]))
			{
				branch_layer_t *me = (branch_layer_t *)n->layer[i];
				for (int j = 0; j < me->num; ++j)
				{
					me->branch[j].n->rate = n->rate;
					net_update(me->branch[j].n);
				}
			}
			else if (is_rnn_layer(n->layer[i]))
			{
				rnn_layer_t *rnn = (rnn_layer_t *)n->layer[i];
				data_update_adam(&n->layer[i]->weight, n->rate / rnn->len, n->t);
				data_update_adam(&n->layer[i]->bias, n->rate / rnn->len, n->t++);
			}
			else
			{
				data_update_adam(&n->layer[i]->weight, n->rate, n->t);
				data_update_adam(&n->layer[i]->bias, n->rate, n->t++);
			}
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

void net_train(net_t *n, feed_func_t feed, data_val_t rate)
{
	int i = 0, b = 0;

	n->train = 1;
	n->rate = rate / n->batch;

	feed(n);

	net_forward(n);

	for (b = 0; b < n->batch; ++b)
		for (i = 0; i < LAST_LAYER(n)->out.size; ++i)
		{
			LAST_LAYER(n)->out.grad[b * LAST_LAYER(n)->out.size + i] = 1;
		}

	net_backward(n);

	net_update(n);

	n->train = 0;
}

static void net_param_save_imp(net_t *n, FILE *fp)
{
	for (int i = 0; i < n->size; ++i)
	{
		if (!is_branch_layer(n->layer[i]))
		{
			data_save(&n->layer[i]->weight, fp);
			data_save(&n->layer[i]->bias, fp);
		}
		else
		{
			branch_layer_t *me = (branch_layer_t *)n->layer[i];
			for (int j = 0; j < me->num; ++j)
			{
				net_param_save_imp(me->branch[j].n, fp);
			}
		}
	}
}

void net_param_save(net_t *n, const char *file)
{
	FILE *fp = fopen(file, "wb");

	net_param_save_imp(n, fp);

	fclose(fp);
}

static void net_param_load_imp(net_t *n, FILE *fp)
{
	for (int i = 0; i < n->size; ++i)
	{
		if (!is_branch_layer(n->layer[i]))
		{
			data_load(fp, &n->layer[i]->weight);
			data_load(fp, &n->layer[i]->bias);
		}
		else
		{
			branch_layer_t *me = (branch_layer_t *)n->layer[i];
			for (int j = 0; j < me->num; ++j)
			{
				net_param_load_imp(me->branch[j].n, fp);
			}
		}
	}
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

	net_param_load_imp(n, fp);

	fclose(fp);
}
