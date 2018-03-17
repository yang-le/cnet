#include <string.h>
#include <time.h>
#ifdef USE_OPENCL
#include "clhelper.h"
#endif
#ifdef USE_CUDA
#include "cudahelper.h"
#endif
#include "mnist.h"
#include "net.h"
#include "layers.h"
#include "log.h"
#include "inception.h"

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

	NET_CREATE(n, TRAIN_SGD, 10);

	NET_ADD(n, conv_layer(3, 224, 224, 64, 112, 112, 7, 2, 3, FILLER_XAVIER, 0.5, 0));
	NET_ADD(n, relu_layer(0));
	NET_ADD(n, max_pooling_layer(64, 112, 112, 56, 56, 3, 2, 0));

	NET_ADD(n, conv_layer(64, 56, 56, 192, 56, 56, 3, 1, 0, FILLER_XAVIER, 0.5, 0));
	NET_ADD(n, relu_layer(0));
	NET_ADD(n, max_pooling_layer(192, 56, 56, 28, 28, 3, 2, 0));

	NET_ADD_INCEPTION_V1(n, 192, 28, 28, 64, 96, 128, 16, 32, 32, FILLER_XAVIER, 0.5, 0);
	NET_ADD_INCEPTION_V1(n, 256, 28, 28, 128, 128, 192, 32, 96, 64, FILLER_XAVIER, 0.5, 0);
	NET_ADD(n, max_pooling_layer(480, 28, 28, 14, 14, 3, 2, 0));

	NET_ADD_INCEPTION_V1(n, 480, 14, 14, 192, 96, 208, 16, 48, 64, FILLER_XAVIER, 0.5, 0);
	NET_ADD_INCEPTION_V1(n, 512, 14, 14, 160, 112, 224, 24, 64, 64, FILLER_XAVIER, 0.5, 0);
	NET_ADD_INCEPTION_V1(n, 512, 14, 14, 128, 128, 256, 24, 64, 64, FILLER_XAVIER, 0.5, 0);
	NET_ADD_INCEPTION_V1(n, 512, 14, 14, 112, 144, 288, 32, 64, 64, FILLER_XAVIER, 0.5, 0);
	NET_ADD_INCEPTION_V1(n, 528, 14, 14, 256, 160, 320, 32, 128, 128, FILLER_XAVIER, 0.5, 0);
	NET_ADD(n, max_pooling_layer(832, 14, 14, 7, 7, 3, 2, 0));

	NET_ADD_INCEPTION_V1(n, 832, 7, 7, 256, 160, 320, 32, 128, 128, FILLER_XAVIER, 0.5, 0);
	NET_ADD_INCEPTION_V1(n, 832, 7, 7, 384, 192, 384, 48, 128, 128, FILLER_XAVIER, 0.5, 0);
	NET_ADD(n, avg_pooling_layer(1024, 7, 7, 1, 1, 7, 1, 0));
	NET_ADD(n, dropout_layer(0, 0.6));

	NET_ADD(n, fc_layer(0, 1000, FILLER_XAVIER, 0.5, 0));
	NET_ADD(n, softmax_layer(0));
	NET_ADD(n, cee_layer(0));

	NET_FINISH(n);

	images = mnist_open(argv[1]);
	labels = mnist_open(argv[2]);

	net_param_load(n, "params.bin");

	for (int i = 0; i < 10000; ++i)
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
					//cvRectangle(&M[0], cvPoint(0, b * 28), cvPoint(27, b * 28 + 27), cvScalar(255, 255, 255, 255), 1, 8, 0);
					LOG("%d ", predict);
				}
			}
			LOG("\n");
			CV_DATA_SHOW_VAL("input", 100, &n->layer[0]->in, 0, 28, 10 * 28, 28, 10 * 28);
			CV_DATA_SHOW_VAL("conv_1", 100, &n->layer[0]->out, 0, 28, 10 * 28, 28, 10 * 28);
			CV_DATA_SHOW_VAL("conv_2", 100, &n->layer[3]->out, 0, 14, 10 * 14, 28, 10 * 28);
			CV_DATA_SHOW_VAL("v1_in", 100, &n->layer[6]->in, 0, 7, 10 * 7, 28, 10 * 28);
			CV_DATA_SHOW_VAL("v1_out", 100, &n->layer[7]->out, 0, 7, 10 * 7, 28, 10 * 28);
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
