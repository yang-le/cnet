#pragma once

#include <stdlib.h>
#include "layer.h"

#define FORWARD_ONLY	0
#define TRAIN_DEFAULT	1
#define TRAIN_MOMENT	2
#define TRAIN_ADAM		3

typedef struct {
	layer_t **layer;
	int size;
	float rate;
} net_t;

typedef void feed_func_t(net_t *n);

//#define FIRST_LAYER(n) ((n)->layer[0])
#define LAST_LAYER(n) ((n)->layer[(n)->size - 1])

net_t* net_create(size_t size);
void net_add(net_t *n, layer_t *l);
void net_finish(net_t *n, int level);
void net_destroy(net_t *n);

void net_forward(net_t *n);
void net_backward(net_t *n);
void net_update(net_t *n);

void net_train(net_t *n, feed_func_t feed, float rate, int round);
