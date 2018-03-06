#pragma once

#include <stdio.h>
#include <stdlib.h>
#ifdef USE_OPENCL
#include "clhelper.h"
#endif
#ifdef USE_CUDA
#include "cudahelper.h"
#endif

typedef float data_val_t;

#ifdef USE_OPENCL
typedef struct
{
	cl_mem buf;
	void *p;
	int size;
} cl_data_val_t;
#endif

typedef struct
{
	data_val_t *val;
#ifdef USE_CUDA
	data_val_t *cuval;
#elif defined(USE_OPENCL)
	cl_data_val_t clval;
#endif
	data_val_t *grad;
#ifdef USE_CUDA
	data_val_t *cugrad;
#elif defined(USE_OPENCL)
	cl_data_val_t clgrad;
#endif
	data_val_t *m; // for moment & adam
	data_val_t *v; // for adam

	int size;
} data_t;

size_t data_init(data_t *data, data_val_t *buf, int level, int batch);
void data_update(data_t *data, double rate);
void data_update_adam(data_t *data);
void data_load(FILE *fp, data_t *data);
void data_save(const data_t *data, FILE *fp);

#ifdef USE_OPENCL
void cl_data_map(cl_data_val_t *data);
void cl_data_unmap(cl_data_val_t *data);
#endif