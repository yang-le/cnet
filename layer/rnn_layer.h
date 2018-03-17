#pragma once

#include "layer.h"

typedef struct
{
    layer_t l;

    int len;
} rnn_layer_t;

#ifdef __cplusplus
extern "C" {
#endif

layer_t *rnn_layer(int in, int out, int len, int filler, float p0, float p1);
int is_rnn_layer(layer_t *l);

#ifdef __cplusplus
}
#endif
