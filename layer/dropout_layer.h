#pragma once

#include "layer.h"

typedef struct
{
	layer_t l;
	float prob;
} dropout_layer_t;

#ifdef __cplusplus
extern "C" {
#endif

layer_t *dropout_layer(int n, float keeprob);

#ifdef __cplusplus
}
#endif
