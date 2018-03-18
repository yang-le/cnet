#include "rnn_layer.h"
#include "log.h"
#include "gemm.h"
#include "common.h"
#include <memory.h>
#include <math.h>

static void rnn_layer_prepare(layer_t *l)
{
    rnn_layer_t *rnn = (rnn_layer_t *)l;

    l->extra.size *= ((l->n->method != TRAIN_FORWARD) + 1) * l->n->batch;
    l->weight.size = ((l->in.size + l->out.size) / rnn->len + rnn->state) * rnn->state;

    LOG("rnn_layer: (%d => %d => %d) x %d, param %d\n",
        l->in.size / rnn->len, rnn->state, l->out.size / rnn->len, rnn->len, l->weight.size + l->bias.size);
}

static void rnn_layer_forward(layer_t *l)
{
    int i = 0;
    int b = 0;
    int t = 0;

    rnn_layer_t *rnn = (rnn_layer_t *)l;

    int batch = l->n->batch;
    int in = l->in.size / rnn->len;
    int out = l->out.size / rnn->len;
    int self = rnn->state;

    int extra_weight_offset = in * self;
    int out_weight_offset = extra_weight_offset + self * self;
    int prev_extra_offset = self;

    for (t = 0; t < rnn->len; ++t)
    {
        int in_offset = t * batch * in;
        int out_offset = t * batch * out;

        // extra hold prev state
        memcpy(l->extra.val + prev_extra_offset, l->extra.val, self * sizeof(data_val_t));

        gemm(0, 1, batch, self, in, 1, &l->in.val, in_offset, in, &l->weight.val, 0, in, 0, &l->extra.val, 0, self);
        gemm(0, 1, batch, self, self, 1, &l->extra.val, prev_extra_offset, self, &l->weight.val, extra_weight_offset, self, 1, &l->extra.val, 0, self);

        for (b = 0; b < batch; ++b)
            for (i = 0; i < self; ++i)
            {
                l->extra.val[b * self + i] = tanh(l->extra.val[b * self + i] + l->bias.val[i]);
            }

        gemm(0, 1, batch, out, self, 1, &l->extra.val, 0, self, &l->weight.val, out_weight_offset, self, 0, &l->out.val, out_offset, out);

        for (b = 0; b < batch; ++b)
            for (i = 0; i < out; ++i)
            {
                l->out.val[out_offset + b * out + i] += l->bias.val[self + i];
            }
    }
}

static void rnn_layer_backward(layer_t *l)
{
    int i = 0;
    int b = 0;
    int t = 0;

    rnn_layer_t *rnn = (rnn_layer_t *)l;

    int batch = l->n->batch;
    int in = l->in.size / rnn->len;
    int out = l->out.size / rnn->len;
    int self = rnn->state;

    int extra_weight_offset = in * self;
    int extra_grad_offset = 2 * self;
    int out_weight_offset = extra_weight_offset + self * self;
    int prev_extra_offset = self;
    int prev_extra_grad_offset = extra_grad_offset + prev_extra_offset;

    for (t = rnn->len - 1; t >= 0; --t)
    {
        int in_offset = t * batch * in;
        int out_offset = t * batch * out;

        gemm(1, 0, out, self, batch, 1, &l->out.grad, out_offset, out, &l->extra.val, 0, self, 0, &l->weight.grad, out_weight_offset, self);
        gemm(0, 0, batch, self, out, 1, &l->out.grad, out_offset, out, &l->weight.val, out_weight_offset, self, 0, &l->extra.grad, extra_grad_offset, self);

        for (b = 0; b < batch; ++b)
            for (i = 0; i < out; ++i)
            {
                l->bias.grad[self + i] += l->out.grad[out_offset + b * out + i];
            }

        // bp through tanh
        for (b = 0; b < batch; ++b)
            for (i = 0; i < self; ++i)
            {
                l->extra.grad[extra_grad_offset + b * self + i] += l->extra.grad[prev_extra_grad_offset + b * self + i];
                l->extra.grad[extra_grad_offset + b * self + i] *= 1 - l->extra.val[b * self + i] * l->extra.val[b * self + i];
            }

        gemm(1, 0, self, in, batch, 1, &l->extra.grad, 0, self, &l->in.val, in_offset, in, 0, &l->weight.grad, 0, in);
        gemm(1, 0, self, self, batch, 1, &l->extra.grad, 0, self, &l->extra.val, prev_extra_offset, self, 0, &l->weight.grad, extra_weight_offset, self);

        gemm(0, 0, batch, in, self, 1, &l->extra.grad, 0, self, &l->weight.val, 0, in, 0, &l->in.grad, in_offset, in);
        gemm(0, 0, batch, self, self, 1, &l->extra.grad, 0, self, &l->weight.val, extra_weight_offset, self, 0, &l->extra.grad, prev_extra_grad_offset, self);

        for (b = 0; b < batch; ++b)
            for (i = 0; i < self; ++i)
            {
                l->bias.grad[i] += l->extra.grad[b * self + i];
            }
    }
}

static const layer_func_t rnn_func = {
    rnn_layer_prepare,
    rnn_layer_forward,
    rnn_layer_backward};

layer_t *rnn_layer(int in, int out, int state, int len, int filler, float p0, float p1)
{
    rnn_layer_t *rnn = (rnn_layer_t *)alloc(1, sizeof(rnn_layer_t));

    rnn->state = state;
    rnn->len = len;

    rnn->l.in.size = in * len;
    rnn->l.out.size = out * len;
    rnn->l.extra.size = 2 * state;
    rnn->l.bias.size = state + out;

    rnn->l.func = &rnn_func;

    layer_set_filler(&rnn->l.weight_filler, filler, p0, p1);

    return (layer_t *)rnn;
}

int is_rnn_layer(layer_t *l)
{
    return l->func == &rnn_func;
}
