#pragma once

#include "layer.h"
#include "branch_layer.h"
#include <stdarg.h>

typedef branch_layer_t merge_layer_t;

#ifdef __cplusplus
extern "C" {
#endif

#define MERGE_LAYER(in, n, offset, ...) merge_layer(in, n, offset, ##__VA_ARGS__, NULL)

layer_t *merge_layer(int out, net_t *n, int offset, ...);

#ifdef __cplusplus
}
#endif
