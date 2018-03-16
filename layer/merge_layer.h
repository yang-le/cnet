#pragma once

#include "layer.h"
#include "branch_layer.h"
#include <stdarg.h>

typedef branch_layer_t merge_layer_t;

layer_t *merge_layer(int out, net_t *n, int offset, ...);
