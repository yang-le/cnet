#include "normalization_layer.h"
#include "common.h"
#include "log.h"
#include <math.h>
#include <memory.h>

static void norm_layer_prepare(layer_t *l)
{
    if (l->in.size == 0)
    {
        l->in.size = l->out.size;
    }

    if (l->out.size == 0)
    {
        l->out.size = l->in.size;
    }

    LOG("normalization_layer: in %d\n", l->in.size);
}

static void bn_layer_prepare(layer_t *l)
{
    l->extra.size = 2 * l->in.size;
    norm_layer_prepare(l);
}

static void bn_layer_forward(layer_t *l)
{
    const double EPSILON = 1e-8;
    int i = 0, b = 0;

    memset(l->extra.val, 0, l->extra.size);
    for (b = 0; b < l->n->batch; ++b)
    {
        for (i = 0; i < l->in.size; ++i)
        {
            l->extra.val[i] += l->in.val[b * l->in.size + i];
        }
    }

    for (i = 0; i < l->in.size; ++i)
    {
        l->extra.val[i] /= l->n->batch;
    }

    for (b = 0; b < l->n->batch; ++b)
    {
        for (i = 0; i < l->in.size; ++i)
        {
            l->extra.val[l->in.size + i] +=
                (l->in.val[b * l->in.size + i] - l->extra.val[i]) * (l->in.val[b * l->in.size + i] - l->extra.val[i]);
        }
    }

    for (i = 0; i < l->in.size; ++i)
    {
        l->extra.val[l->in.size + i] = sqrt(l->extra.val[l->in.size + i] / l->n->batch + EPSILON);
    }

    for (b = 0; b < l->n->batch; ++b)
    {
        for (i = 0; i < l->out.size; ++i)
        {
            l->out.val[b * l->out.size + i] = (l->in.val[b * l->in.size + i] - l->extra.val[i]) / l->extra.val[l->in.size + i];
        }
    }
}

static void bn_layer_backward(layer_t *l)
{
    int i = 0, b = 0;

    for (b = 0; b < l->n->batch; ++b)
    {
        for (i = 0; i < l->in.size; ++i)
        {
            l->in.grad[b * l->in.size + i] = l->out.grad[b * l->out.size + i] / l->extra.val[l->in.size + i];
        }
    }
}

static const layer_func_t bn_func = {
    bn_layer_prepare,
    bn_layer_forward,
    bn_layer_backward};

layer_t *bn_layer(int in)
{
    return layer(in, in, 0, 0, 0, &bn_func);
}
