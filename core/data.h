#pragma once

#include <stdio.h>
#include <stdlib.h>
#ifdef USE_OPENCL
#include "clhelper.h"
#endif
#ifdef USE_CUDA
#include "cudahelper.h"
#endif
#ifdef USE_OPENCV
#include <cv.h>
#include <highgui.h>
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
#ifdef USE_OPENCV
	CvMat *cvval;
#endif
	data_val_t *grad;
#ifdef USE_CUDA
	data_val_t *cugrad;
#elif defined(USE_OPENCL)
	cl_data_val_t clgrad;
#endif
#ifdef USE_OPENCV
	CvMat *cvgrad;
#endif
	data_val_t *m; // for moment & adam
	data_val_t *v; // for adam

	int size;
} data_t;

#define DEFAULT_ADAM_RATE 0.001

size_t data_init(data_t *data, data_val_t *buf, int level, int batch);

void data_update_nesterov(data_t *data);
void data_update_sgd(data_t *data, double rate);
void data_update_momentum(data_t *data, double rate);
void data_update_adagrad(data_t *data, double rate);
void data_update_adadelta(data_t *data, double rate);
void data_update_adam(data_t *data, double rate, int t);

void data_load(FILE *fp, data_t *data);
void data_save(const data_t *data, FILE *fp);

#ifdef USE_OPENCL
void cl_data_map(cl_data_val_t *data);
void cl_data_unmap(cl_data_val_t *data);
#endif

#ifdef USE_OPENCV
void cv_data_show(char *window, int delay, data_t *data, CvMat *cvdata, int offset, int iw, int ih, int ow, int oh);
#define CV_DATA_SHOW_VAL(window, delay, data, offset, iw, ih, ow, oh) cv_data_show(window, delay, data, (data)->cvval, offset, iw, ih, ow, oh)
#define CV_DATA_SHOW_GRAD(window, delay, data, offset, iw, ih, ow, oh) cv_data_show(window, delay, data, (data)->cvgrad, offset, iw, ih, ow, oh)
#endif
