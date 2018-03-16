#pragma once

#include "layer.h"

#ifdef __cplusplus
extern "C" {
#endif

layer_t *fc_layer(int in, int out, int filler, float p0, float p1);

#ifdef __cplusplus
}
#endif
