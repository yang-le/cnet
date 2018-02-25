#pragma once

#include <stdlib.h>

typedef float data_val_t;

typedef struct {
	data_val_t *val;
	data_val_t *grad;
	data_val_t *m;		// for moment & adam
	data_val_t *v;		// for adam
	int immutable;
	int size;
} data_t;

size_t data_init(data_t *data, data_val_t *buf, int level);
void data_update(data_t *data, double rate);
void data_load(char *file, data_t *data);
void data_save(data_t *data, char *file);
