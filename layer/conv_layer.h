#pragma once

#include "layer.h"

typedef struct {
	layer_t l;

	int ic;	// in channels
	int iw;	// in width
	int ih;	// in height

	int oc;	// out channels
	int ow;	// out width
	int oh; // out height

	int k;	// kernel size
	int s;	// step / stride
	int p;	// padding

	data_t col;	// the working space
} conv_layer_t;

layer_t * conv_layer(int ic, int iw, int ih, int oc, int ow, int oh, int k, int s, int p);
