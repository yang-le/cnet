#include "rnn_layer.h"
#include "log.h"
#include "gemm.h"
#include "common.h"
#include <memory.h>

static void rnn_layer_prepare(layer_t *l)
{
    rnn_layer_t *rnn = (rnn_layer_t *)l;

    l->extra.size *= ((l->n->method != TRAIN_FORWARD) + 1) * l->n->batch;
    l->weight.size = (l->in.size + l->out.size) / rnn->len * l->out.size / rnn->len;

    LOG("rnn_layer: in %d, out %d, param %d\n", l->in.size, l->out.size, l->weight.size + l->bias.size);
}

static void rnn_layer_forward(layer_t *l)
{
    int i = 0;
    int b = 0;
    int t = 0;

    rnn_layer_t *rnn = (rnn_layer_t *)l;

    int m = l->n->batch;
    int k = l->in.size / rnn->len;
    int n = l->out.size / rnn->len;

    for (t = 0; t < rnn->len; ++t)
    {
        int in_offset = t * m * k;
        int out_offset = t * m * n;
        int extra_offset = out_offset;

        // extra hold prev state
        memcpy(l->extra.val, l->out.val, rnn->len * m * n * sizeof(data_val_t));

        gemm(0, 1, m, n, k, 1, &l->in.val, in_offset, k, &l->weight.val, 0, k, 0, &l->out.val, out_offset, n);
        gemm(0, 1, m, n, n, 1, &l->extra.val, extra_offset, n, &l->weight.val, k * n, n, 1, &l->out.val, out_offset, n);

        for (b = 0; b < m; ++b)
            for (i = 0; i < n; ++i)
            {
                l->out.val[out_offset + b * n + i] += l->bias.val[i];
            }
    }
}

static void rnn_layer_backward(layer_t *l)
{
    int i = 0;
    int b = 0;
    int t = 0;

    rnn_layer_t *rnn = (rnn_layer_t *)l;

    for (t = rnn->len - 1; t >= 0; --t)
    {
        int m = l->out.size / rnn->len;
        int k = l->n->batch;
        int n = l->in.size / rnn->len;

        int in_offset = t * k * n;
        int out_offset = t * k * m;
        int extra_val_offset = out_offset;
        int extra_grad_offset = rnn->len * k * m + extra_val_offset;

        // add prev state's grad
        for (b = 0; b < k; ++b)
            for (i = 0; i < m; ++i)
            {
                l->out.grad[out_offset + b * m + i] += l->extra.grad[extra_grad_offset + b * m + i];
            }

        gemm(1, 0, m, n, k, 1, &l->out.grad, out_offset, m, &l->in.val, in_offset, n, 1, &l->weight.grad, 0, n);
        gemm(1, 0, m, m, k, 1, &l->out.grad, out_offset, m, &l->extra.val, extra_val_offset, m, 1, &l->weight.grad, n * m, m);

        m = l->n->batch;
        k = l->out.size;
        n = l->in.size;

        in_offset = t * m * n;
        out_offset = t * m * k;
        extra_val_offset = out_offset;
        extra_grad_offset = rnn->len * m * k + extra_val_offset;

        gemm(0, 0, m, n, k, 1, &l->out.grad, out_offset, k, &l->weight.val, 0, n, 0, &l->in.grad, in_offset, n);
        gemm(0, 0, m, k, k, 1, &l->out.grad, out_offset, k, &l->weight.val, n * k, k, 0, &l->extra.grad, extra_grad_offset, k);

        for (b = 0; b < m; ++b)
            for (i = 0; i < k; ++i)
            {
                l->bias.grad[i] += l->out.grad[out_offset + b * k + i];
            }
    }
}

static const layer_func_t rnn_func = {
    rnn_layer_prepare,
    rnn_layer_forward,
    rnn_layer_backward};

layer_t *rnn_layer(int in, int out, int len, int filler, float p0, float p1)
{
    rnn_layer_t *rnn = (rnn_layer_t *)alloc(1, sizeof(rnn_layer_t));

    rnn->len = len;

    rnn->l.in.size = in;
    rnn->l.out.size = out;
    rnn->l.weight.size = (in + out) / len * out / len;
    rnn->l.bias.size = out / len;
    rnn->l.extra.size = out;
    rnn->l.func = &rnn_func;

    layer_set_filler(&rnn->l.weight_filler, filler, p0, p1);

    return (layer_t *)rnn;
}

int is_rnn_layer(layer_t *l)
{
    return l->func == &rnn_func;
}
