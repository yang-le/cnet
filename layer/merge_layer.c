#include "common.h"
#include "log.h"
#include "merge_layer.h"
#include <memory.h>

static void merge_layer_prepare(layer_t *l)
{
    LOG("merge_layer: out %d\n", l->out.size);
}

static void merge_layer_forward(layer_t *l)
{
    int i = 0, j = 0, b = 0;
    merge_layer_t *me = (merge_layer_t *)l;

    memset(l->out.val, 0, l->n->batch * l->out.size * sizeof(data_val_t));

    for (i = 0; i < me->num; ++i)
    {
        layer_t *branch = LAST_LAYER(me->branch[i].n);
        for (b = 0; b < l->n->batch; ++b)
            for (j = 0; j < branch->out.size; ++j)
                l->out.val[b * l->out.size + me->branch[i].offset + j] += branch->out.val[b * branch->out.size + j];
    }
}

static void merge_layer_backward(layer_t *l)
{
    int i = 0, b = 0;
    merge_layer_t *me = (merge_layer_t *)l;

    for (i = 0; i < me->num; ++i)
    {
        layer_t *branch = LAST_LAYER(me->branch[i].n);
        for (b = 0; b < l->n->batch; ++b)
            memcpy(&branch->out.grad[b * branch->out.size], &l->out.grad[b * l->out.size + me->branch[i].offset],
                   branch->out.size * sizeof(data_val_t));

        net_backward(me->branch[i].n);
    }
}

static const layer_func_t merge_func = {
    merge_layer_prepare,
    merge_layer_forward,
    merge_layer_backward};

layer_t *merge_layer(int out, net_t *n, int offset, ...)
{
    int num = 1;
    va_list branches;

    va_start(branches, offset);

    for (net_t *_n = NULL; _n = va_arg(branches, net_t *); ++num)
    {
        va_arg(branches, int);
    }

    va_end(branches);

    merge_layer_t *me = (merge_layer_t *)alloc(1, sizeof(merge_layer_t) + num * sizeof(branch_t));

    me->l.out.size = out;
    me->l.func = &merge_func;

    me->num = 1;
    me->branch = (branch_t *)(me + 1);

    me->branch[0].n = n;
    me->branch[0].offset = offset;

    va_start(branches, offset);

    for (net_t *_n = NULL; _n = va_arg(branches, net_t *); ++(me->num))
    {
        me->branch[me->num].n = _n;
        me->branch[me->num].offset = va_arg(branches, int);
    }

    va_end(branches);

    return (layer_t *)me;
}
