#pragma once

#include "layer.h"

typedef struct
{
	layer_t l;
	float prob;
} dropout_layer_t;

layer_t *dropout_layer(int n, float keeprob);
