#include <string.h>
#include <time.h>
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
#include "mnist.h"
#include "net.h"
#include "log.h"
#include "fc_layer.h"
#include "softmax_layer.h"
#include "cee_layer.h"
#include "conv_layer.h"
#include "pooling_layer.h"
#include "relu_layer.h"
#include "dropout_layer.h"
#include "normalization_layer.h"
#include "scale_layer.h"

static idx_t *images = NULL;
static idx_t *labels = NULL;

static void feed_data(net_t *n)
{
	for (int b = 0; b < n->batch; ++b)
	{
		int i = rand();

		int j = 0;
		for (j = 0; j < 28 * 28; ++j)
			n->layer[0]->in.val[b * 28 * 28 + j] = images->data[(i % images->dim[0]) * 28 * 28 + j];

		for (j = 0; j < 10; ++j)
			LAST_LAYER(n)->extra.val[b * 10 + j] = (labels->data[i % labels->dim[0]] == j);
	}
}

static int arg_max(data_val_t *data, int n)
{
	int i = 0;
	int max = 0;

	for (i = 0; i < n; ++i)
		if (data[i] > data[max])
			max = i;

	return max;
}

int main(int argc, char **argv)
{
	int right = 0;
	float loss = 0;
	time_t start = time(NULL);
	net_t *n = NULL;

#ifdef USE_OPENCL
	cl_init();
#endif
#ifdef USE_CUDA
	cublas_init();
#endif

	NET_CREATE(n, TRAIN_SGD, 100);

	NET_ADD(n, conv_layer(1, 28, 28, 32, 28, 28, 5, 0, 0, FILLER_MSRA, 0.5, 0));
	NET_ADD(n, relu_layer(0));
	NET_ADD(n, max_pooling_layer(32, 28, 28, 14, 14, 2, 0, 0));

	NET_ADD(n, conv_layer(32, 14, 14, 64, 14, 14, 5, 0, 0, FILLER_MSRA, 0.5, 0));
	NET_ADD(n, relu_layer(0));
	NET_ADD(n, max_pooling_layer(64, 14, 14, 7, 7, 2, 0, 0));

	NET_ADD(n, fc_layer(0, 1024, FILLER_MSRA, 0.5, 0));
	NET_ADD(n, relu_layer(0));
	NET_ADD(n, dropout_layer(0, 0.6));

	NET_ADD(n, fc_layer(0, 10, FILLER_MSRA, 0.5, 0));
	NET_ADD(n, softmax_layer(0));
	NET_ADD(n, cee_layer(0));

	NET_FINISH(n);

#ifdef USE_OPENCV
	CvMat M[10];
	cvInitMatHeader(&M[0], 10 * 28, 28, CV_32FC1, n->layer[0]->in.val, 28 * sizeof(data_val_t));
	cvNamedWindow("input", CV_WINDOW_AUTOSIZE);
	cvInitMatHeader(&M[1], 10 * 28, 28, CV_32FC1, n->layer[1]->in.val, 28 * sizeof(data_val_t));
	cvNamedWindow("conv1", CV_WINDOW_AUTOSIZE);
	cvInitMatHeader(&M[2], 10 * 28, 28, CV_32FC1, n->layer[2]->in.val, 28 * sizeof(data_val_t));
	cvNamedWindow("relu1", CV_WINDOW_AUTOSIZE);
	cvInitMatHeader(&M[3], 10 * 14, 14, CV_32FC1, n->layer[3]->in.val, 14 * sizeof(data_val_t));
	cvNamedWindow("pool1", CV_WINDOW_AUTOSIZE);
	cvInitMatHeader(&M[4], 10 * 14, 14, CV_32FC1, n->layer[4]->in.val, 14 * sizeof(data_val_t));
	cvNamedWindow("conv2", CV_WINDOW_AUTOSIZE);
	cvInitMatHeader(&M[5], 10 * 14, 14, CV_32FC1, n->layer[5]->in.val, 14 * sizeof(data_val_t));
	cvNamedWindow("relu2", CV_WINDOW_AUTOSIZE);
	cvInitMatHeader(&M[6], 10 * 7, 7, CV_32FC1, n->layer[6]->in.val, 7 * sizeof(data_val_t));
	cvNamedWindow("pool2", CV_WINDOW_AUTOSIZE);
#endif

	images = mnist_open(argv[1]);
	labels = mnist_open(argv[2]);

	net_param_load(n, "params.bin");

	for (int i = 0; i < 20000; ++i)
	{
		net_train(n, feed_data, 0.001);

		for (int b = 0; b < n->batch; ++b)
			loss += LAST_LAYER(n)->out.val[b];

		if (i % 50 == 0)
		{
			feed_data(n);
			net_forward(n);
#ifdef USE_OPENCV
			for (int b = 0; b < 10; ++b)
			{
				int predict = arg_max(&LAST_LAYER(n)->in.val[b * 10], 10);
				int truth = arg_max(&LAST_LAYER(n)->extra.val[b * 10], 10);
				if (predict != truth)
				{
					cvRectangle(&M[0], cvPoint(0, b * 28), cvPoint(27, b * 28 + 27), cvScalar(255, 255, 255, 255), 1, 8, 0);
					LOG("%d ", predict);
				}
			}
			LOG("\n");
			cvShowImage("input", &M[0]);
			cvShowImage("conv1", &M[1]);
			cvShowImage("relu1", &M[2]);
			cvShowImage("pool1", &M[3]);
			cvShowImage("conv2", &M[4]);
			cvShowImage("relu2", &M[5]);
			cvShowImage("pool2", &M[6]);
			cvWaitKey(200);
#endif
			for (int b = 0; b < n->batch; ++b)
				right += (arg_max(&LAST_LAYER(n)->in.val[b * 10], 10) == arg_max(&LAST_LAYER(n)->extra.val[b * 10], 10));

			LOG("loss = %f, train accurcy = %f, step = %d\n", loss / n->batch / 50, 1.0 * right / n->batch, i);
			LOG("steps/sec = %f\n", 50.0 / (time(NULL) - start));

			loss = 0;
			right = 0;
			start = time(NULL);

			net_param_save(n, "params.bin");
		}
	}

	net_param_save(n, "params.bin");

	mnist_close(labels);
	mnist_close(images);

	images = mnist_open(argv[3]);
	labels = mnist_open(argv[4]);

	for (int i = 0; i < images->dim[0] / n->batch; ++i)
	{
		feed_data(n);
		net_forward(n);

		for (int b = 0; b < n->batch; ++b)
			right += (arg_max(&LAST_LAYER(n)->in.val[b * 10], 10) == arg_max(&LAST_LAYER(n)->extra.val[b * 10], 10));
	}

	LOG("accurcy %f\n", 1.0 * right / (images->dim[0] / n->batch * n->batch));

	mnist_close(labels);
	mnist_close(images);

	net_destroy(n);
#ifdef USE_CUDA
	cublas_deinit();
#endif
#ifdef USE_OPENCL
	cl_deinit();
#endif
	return 0;
}
