#pragma once

#include "layer.h"
#include <stdarg.h>

typedef struct
{
    int offset;
    net_t *n;
} branch_t;

typedef struct
{
    layer_t l;

    int num;
    branch_t *branch;
} branch_layer_t;

layer_t *branch_layer(int in, net_t *n, int offset, ...);
