#pragma once

#include <stdlib.h>
#include "layer.h"

struct layer;

typedef struct net
{
	struct layer **layer;
	int size;
	int train;
	enum
	{
		TRAIN_FORWARD,
		TRAIN_SGD,
		TRAIN_MOMENTUM,
		TRAIN_NESTEROV,
		TRAIN_ADAGRAD,
		TRAIN_ADADELTA,
		TRAIN_ADAM,
		TRAIN_MAX
	} method;
	int batch;
	int t;
	float rate;
} net_t;

typedef void feed_func_t(net_t *n);

//#define FIRST_LAYER(n) ((n)->layer[0])
#define LAST_LAYER(n) ((n)->layer[(n)->size - 1])

#define NET_CREATE(n, method, batch) \
	{                                \
		int _cnt = 0;                \
		n = NULL;                    \
	_start:                          \
		if (_cnt)                    \
		n = net_create(_cnt, method, batch)

#define NET_ADD(n, l)  \
	if (n)             \
		net_add(n, l); \
	else               \
		++_cnt

#define NET_FINISH(n)  \
	if (n)             \
		net_finish(n); \
	else               \
		goto _start;   \
	}

net_t *net_create(int size, int method, int batch);
void net_add(net_t *n, struct layer *l);
void net_finish(net_t *n);
void net_destroy(net_t *n);

void net_forward(net_t *n);
void net_backward(net_t *n);
void net_update(net_t *n);

void net_train(net_t *n, feed_func_t feed, float rate);

void net_param_save(net_t *n, const char *file);
void net_param_load(net_t *n, const char *file);
