#pragma once

#include "conv_layer.h"

typedef conv_layer_t pooling_layer_t;

#ifdef __cplusplus
extern "C" {
#endif

layer_t *max_pooling_layer(int c, int iw, int ih, int ow, int oh, int k, int s, int p);
layer_t *avg_pooling_layer(int c, int iw, int ih, int ow, int oh, int k, int s, int p);

#ifdef __cplusplus
}
#endif
