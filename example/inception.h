#pragma once

#include "net.h"
#include "layers.h"

#define NET_ADD_INCEPTION_V1(n, method, batch, ic, iw, ih, ow, oh, n1, n3r, n3, n5r, n5, np) \
    {                                                           \
        static net_t *_n1 = NULL, *_n3 = NULL, *_n5 = NULL, *_np = NULL;                       \
        if (!_n1) { \
        NET_CREATE(_n1, method, batch);                    \
        NET_ADD(_n1, conv_layer(ic, iw, ih, n1, ow, oh, 1, 0, 0, FILLER_MSRA, 0.5, 0)); \
        NET_ADD(_n1, relu_layer(0)); \
        NET_FINISH(_n1); \
        } \
        if (!_n3) { \
        NET_CREATE(_n3, method, batch);                    \
        NET_ADD(_n3, conv_layer(ic, iw, ih, n3r, ow, oh, 1, 0, 0, FILLER_MSRA, 0.5, 0)); \
        NET_ADD(_n3, relu_layer(0)); \
        NET_ADD(_n3, conv_layer(n3r, iw, ih, n3, ow, oh, 3, 0, 0, FILLER_MSRA, 0.5, 0)); \
        NET_ADD(_n3, relu_layer(0)); \
        NET_FINISH(_n3); \
        } \
        if (!_n5) { \
        NET_CREATE(_n5, method, batch);                    \
        NET_ADD(_n5, conv_layer(ic, iw, ih, n5r, ow, oh, 1, 0, 0, FILLER_MSRA, 0.5, 0)); \
        NET_ADD(_n5, relu_layer(0)); \
        NET_ADD(_n5, conv_layer(n5r, iw, ih, n5, ow, oh, 5, 0, 0, FILLER_MSRA, 0.5, 0)); \
        NET_ADD(_n5, relu_layer(0)); \
        NET_FINISH(_n5); \
         } \
        if (!_np) { \
        NET_CREATE(_np, method, batch);                    \
        NET_ADD(_np, max_pooling_layer(ic, iw, ih, iw, ih, 3, 1, 1)); \
        NET_ADD(_np, conv_layer(ic, iw, ih, np, ow, oh, 1, 0, 0, FILLER_MSRA, 0.5, 0)); \
        NET_ADD(_np, relu_layer(0)); \
        NET_FINISH(_np); \
        } \
        NET_ADD(n, branch_layer(ic * iw * ih, _n1, 0, _n3, 0, _n5, 0, _np, 0, NULL)); \
        NET_ADD(n, merge_layer((n1 + n3 + n5 + np) * ow * oh, _n1, 0, _n3, n1 * ow * oh, _n5, (n1 + n3) * ow * oh, _np, (n1 + n3 + n5) * ow * oh, NULL)); \
}
