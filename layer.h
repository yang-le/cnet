#pragma once

#include "data.h"

struct layer;

typedef struct {
	void (*prepare)(struct layer *l);
	void (*forward)(struct layer *l);
	void (*backward)(struct layer *l);
} layer_func_t;

typedef struct layer {
	const layer_func_t *func;

	data_t in;
	data_t out;
	data_t param;
} layer_t;

#define PREPARE(l) ((l)->func->prepare((l)))
#define FORWARD(l) ((l)->func->forward((l)))
#define BACKWARD(l) ((l)->func->backward((l)))

enum {
	FC = 0,
	RELU,
	SIGMOID,
	SOFTMAX,
	MSE,
	CEE,
	LAYER_MAX
};

layer_t* layer(int in, int out, int param, const layer_func_t *func);
