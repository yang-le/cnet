#pragma once

typedef float data_val_t;

typedef struct {
	data_val_t *val;
	data_val_t *grad;
	int mutable;
	int size;
} data_t;
