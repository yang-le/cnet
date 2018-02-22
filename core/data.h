#pragma once

typedef float data_val_t;

typedef struct {
	data_val_t *val;
	data_val_t *grad;
	int immutable;
	int size;
} data_t;


void data_update(data_t *data, double rate);
void data_load(char *file, data_t *data);
void data_save(data_t *data, char *file);
