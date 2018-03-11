#include "common.h"
#include "branch_layer.h"

layer_t *branch_layer(int in, layer_t *l, int offset, ...)
{
    int num = 1;
    va_list branches;

    va_start(branches, offset);

    for (layer_t *_l = NULL; _l = va_arg(branches, layer_t *); ++num)
    {
        va_arg(branches, int);
    }

    va_end(branches);

    branch_layer_t *me = (branch_layer_t *)alloc(1, sizeof(branch_layer_t) + num * sizeof(branch_t));

    me->l.in.size = in;
    me->l.out.size = in;

    me->num = 1;
    me->branch = (branch_t *)(me + 1);

    me->branch[0].l = l;
    me->branch[0].offset = offset;

    va_start(branches, offset);

    for (layer_t *_l = NULL; _l = va_arg(branches, layer_t *); ++(me->num))
    {
        me->branch[me->num].l = _l;
        me->branch[me->num].offset = va_arg(branches, int);
    }

    va_end(branches);

    return (layer_t *)me;
}