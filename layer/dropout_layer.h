#pragma once

#include "layer.h"

typedef struct {
	layer_t l;
	float prob;
} dropout_layer_t;

layer_t* dropout_layer(int n, float droprob);

#define SET_DROP_PROB(l, p) (((dropout_layer_t*)(l))->prob = (p))