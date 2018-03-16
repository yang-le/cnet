#pragma once

#include "net.h"
#include "layers.h"

#define NET_ADD_VGG_BLOCK(n, num, w, h, ic, oc)                                   \
    NET_ADD(n, conv_layer(ic, w, h, oc, w, h, 3, 1, 0, FILLER_MSRA, 0.5, 0));     \
    NET_ADD(n, relu_layer(0));                                                    \
    for (int i = 1; i < num; ++i)                                                 \
    {                                                                             \
        NET_ADD(n, conv_layer(oc, w, h, oc, w, h, 3, 1, 0, FILLER_MSRA, 0.5, 0)); \
        NET_ADD(n, relu_layer(0));                                                \
    }                                                                             \
    NET_ADD(n, max_pooling_layer(oc, w, h, w / 2, h / 2, 2, 2, 0))

#define NET_VGG_FINISH(n)                               \
    NET_ADD(n, fc_layer(0, 4096, FILLER_MSRA, 0.5, 0)); \
    NET_ADD(n, relu_layer(0));                          \
    NET_ADD(n, dropout_layer(0, 0.5));                  \
    NET_ADD(n, fc_layer(0, 4096, FILLER_MSRA, 0.5, 0)); \
    NET_ADD(n, relu_layer(0));                          \
    NET_ADD(n, dropout_layer(0, 0.5));                  \
    NET_ADD(n, fc_layer(0, 1000, FILLER_MSRA, 0.5, 0)); \
    NET_ADD(n, relu_layer(0));                          \
    NET_ADD(n, softmax_layer(0));                       \
    NET_ADD(n, cee_layer(0));                           \
    NET_FINISH(n)

#define NET_CREATE_VGG_A(n, train, batch)       \
    NET_CREATE(n, train, batch);                \
    NET_ADD_VGG_BLOCK(n, 1, 224, 224, 3, 64);   \
    NET_ADD_VGG_BLOCK(n, 1, 112, 112, 64, 128); \
    NET_ADD_VGG_BLOCK(n, 2, 56, 56, 128, 256);  \
    NET_ADD_VGG_BLOCK(n, 2, 28, 28, 256, 512);  \
    NET_ADD_VGG_BLOCK(n, 2, 14, 14, 512, 512);  \
    NET_VGG_FINISH(n)

#define NET_CREATE_VGG_B(n, train, batch)       \
    NET_CREATE(n, train, batch);                \
    NET_ADD_VGG_BLOCK(n, 2, 224, 224, 3, 64);   \
    NET_ADD_VGG_BLOCK(n, 2, 112, 112, 64, 128); \
    NET_ADD_VGG_BLOCK(n, 2, 56, 56, 128, 256);  \
    NET_ADD_VGG_BLOCK(n, 2, 28, 28, 256, 512);  \
    NET_ADD_VGG_BLOCK(n, 2, 14, 14, 512, 512);  \
    NET_VGG_FINISH(n)

#define NET_CREATE_VGG_D(n, train, batch)       \
    NET_CREATE(n, train, batch);                \
    NET_ADD_VGG_BLOCK(n, 2, 224, 224, 3, 64);   \
    NET_ADD_VGG_BLOCK(n, 2, 112, 112, 64, 128); \
    NET_ADD_VGG_BLOCK(n, 3, 56, 56, 128, 256);  \
    NET_ADD_VGG_BLOCK(n, 3, 28, 28, 256, 512);  \
    NET_ADD_VGG_BLOCK(n, 3, 14, 14, 512, 512);  \
    NET_VGG_FINISH(n)

#define NET_CREATE_VGG_E(n, train, batch)       \
    NET_CREATE(n, train, batch);                \
    NET_ADD_VGG_BLOCK(n, 2, 224, 224, 3, 64);   \
    NET_ADD_VGG_BLOCK(n, 2, 112, 112, 64, 128); \
    NET_ADD_VGG_BLOCK(n, 4, 56, 56, 128, 256);  \
    NET_ADD_VGG_BLOCK(n, 4, 28, 28, 256, 512);  \
    NET_ADD_VGG_BLOCK(n, 4, 14, 14, 512, 512);  \
    NET_VGG_FINISH(n)
