#pragma once

#include "data.h"
#include "net.h"

typedef struct
{
	enum
	{
		FILLER_CONST,
		FILLER_GAUSS,
		FILLER_UNIFORM,
		FILLER_XAVIER,
		FILLER_MSRA
	} method;
	data_val_t param[2];
} data_filler_t;

struct layer;
struct net;

typedef struct
{
	void (*prepare)(struct layer *l);
	void (*forward)(struct layer *l);
	void (*backward)(struct layer *l);
} layer_func_t;

typedef struct layer
{
	const layer_func_t *func;

	data_t in;
	data_t out;

	data_t weight;
	data_t bias;
	data_t extra;

	data_filler_t weight_filler;
	data_filler_t bias_filler;

	struct net *n;
} layer_t;

#define PREPARE(l) ((l)->func->prepare((l)))
#define FORWARD(l) ((l)->func->forward((l)))
#define BACKWARD(l) ((l)->func->backward((l)))

layer_t *layer(int in, int out, int weight, int bias, int extra, const layer_func_t *func);
size_t layer_data_init(layer_t *l, data_val_t *buf);

void layer_set_weight_filler(layer_t *l, int method, data_val_t p1, data_val_t p2);
void layer_set_bias_filler(layer_t *l, int method, data_val_t p1, data_val_t p2);
