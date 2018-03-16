#include "common.h"
#include "log.h"
#include "branch_layer.h"
#include <memory.h>

static void branch_layer_prepare(layer_t *l)
{
    LOG("branch_layer: in %d\n", l->in.size);
}

static void branch_layer_forward(layer_t *l)
{
    int i = 0, b = 0;
    branch_layer_t *me = (branch_layer_t *)l;

    for (i = 0; i < me->num; ++i)
    {
        layer_t *branch = me->branch[i].n->layer[0];
        for (b = 0; b < l->n->batch; ++b)
            memcpy(&branch->in.val[b * branch->in.size], &l->in.val[b * l->in.size + me->branch[i].offset],
                   branch->in.size * sizeof(data_val_t));

        net_forward(me->branch[i].n);
    }
}

static void branch_layer_backward(layer_t *l)
{
    int i = 0, j = 0, b = 0;
    branch_layer_t *me = (branch_layer_t *)l;

    memset(l->in.grad, 0, l->n->batch * l->in.size * sizeof(data_val_t));

    for (i = 0; i < me->num; ++i)
    {
        layer_t *branch = me->branch[i].n->layer[0];
        for (b = 0; b < l->n->batch; ++b)
            for (j = 0; j < branch->in.size; ++j)
                l->in.grad[b * l->in.size + me->branch[i].offset + j] += branch->in.grad[b * branch->in.size + j];
    }
}

static const layer_func_t branch_func = {
    branch_layer_prepare,
    branch_layer_forward,
    branch_layer_backward};

layer_t *branch_layer(int in, net_t *n, int offset, ...)
{
    int num = 1;
    va_list branches;

    va_start(branches, offset);

    for (net_t *_n = NULL; _n = va_arg(branches, net_t *); ++num)
    {
        va_arg(branches, int);
    }

    va_end(branches);

    branch_layer_t *me = (branch_layer_t *)alloc(1, sizeof(branch_layer_t) + num * sizeof(branch_t));

    me->l.in.size = in;
    me->l.func = &branch_func;

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

int is_branch_layer(layer_t *l)
{
    return l->func == &branch_func;
}
