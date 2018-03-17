#pragma once

#include "net.h"
#include "layers.h"

#define NET_ADD_INCEPTION_V1(n, ic, w, h, n1, n3r, n3, n5r, n5, np, filler, p0, p1)                                                               \
    {                                                                                                                                             \
        static net_t *_n1 = NULL, *_n3 = NULL, *_n5 = NULL, *_np = NULL;                                                                          \
        if (n)                                                                                                                                    \
        {                                                                                                                                         \
            NET_CREATE(_n1, n->method, n->batch);                                                                                                 \
            NET_ADD(_n1, conv_layer(ic, w, h, n1, w, h, 1, 0, 0, filler, p0, p1));                                                                \
            NET_ADD(_n1, relu_layer(0));                                                                                                          \
            NET_FINISH(_n1);                                                                                                                      \
            NET_CREATE(_n3, n->method, n->batch);                                                                                                 \
            NET_ADD(_n3, conv_layer(ic, w, h, n3r, w, h, 1, 0, 0, filler, p0, p1));                                                               \
            NET_ADD(_n3, relu_layer(0));                                                                                                          \
            NET_ADD(_n3, conv_layer(n3r, w, h, n3, w, h, 3, 0, 0, filler, p0, p1));                                                               \
            NET_ADD(_n3, relu_layer(0));                                                                                                          \
            NET_FINISH(_n3);                                                                                                                      \
            NET_CREATE(_n5, n->method, n->batch);                                                                                                 \
            NET_ADD(_n5, conv_layer(ic, w, h, n5r, w, h, 1, 0, 0, filler, p0, p1));                                                               \
            NET_ADD(_n5, relu_layer(0));                                                                                                          \
            NET_ADD(_n5, conv_layer(n5r, w, h, n5, w, h, 5, 0, 0, filler, p0, p1));                                                               \
            NET_ADD(_n5, relu_layer(0));                                                                                                          \
            NET_FINISH(_n5);                                                                                                                      \
            NET_CREATE(_np, n->method, n->batch);                                                                                                 \
            NET_ADD(_np, max_pooling_layer(ic, w, h, w, h, 3, 1, 1));                                                                             \
            NET_ADD(_np, conv_layer(ic, w, h, np, w, h, 1, 0, 0, filler, p0, p1));                                                                \
            NET_ADD(_np, relu_layer(0));                                                                                                          \
            NET_FINISH(_np);                                                                                                                      \
        }                                                                                                                                         \
        NET_ADD(n, branch_layer(ic *w *h, _n1, 0, _n3, 0, _n5, 0, _np, 0, NULL));                                                                 \
        NET_ADD(n, merge_layer((n1 + n3 + n5 + np) * w * h, _n1, 0, _n3, n1 * w * h, _n5, (n1 + n3) * w * h, _np, (n1 + n3 + n5) * w * h, NULL)); \
    }
