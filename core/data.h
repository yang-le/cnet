#pragma once

#include <stdio.h>
#include <stdlib.h>
#ifdef USE_OPENCL
#include <CL/cl.h>
#endif

typedef float data_val_t;

typedef struct
{
	data_val_t *val;
	data_val_t *grad;
	data_val_t *m; // for moment & adam
	data_val_t *v; // for adam
#ifdef USE_OPENCL
	cl_mem val_cl;
	cl_mem grad_cl;
#endif
	int size;
	int immutable;
} data_t;

size_t data_init(data_t *data, data_val_t *buf, int level);
void data_update(data_t *data, double rate);
void data_update_adam(data_t *data);
void data_load(FILE *fp, data_t *data);
void data_save(const data_t *data, FILE *fp);
