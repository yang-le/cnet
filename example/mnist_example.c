#include <string.h>
#include <time.h>
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

static idx_t *images = NULL;
static idx_t *labels = NULL;

static void feed_data(net_t *n)
{
	static int i = 0;

	int j = 0;
	for (j = 0; j < 28 * 28; ++j)
		n->layer[0]->in.val[j] = 1.0 * images->data[(i % images->dim[0]) * 28 * 28 + j] / 255;

	for (j = 0; j < 10; ++j)
		LAST_LAYER(n)->param.val[j] = (labels->data[i % labels->dim[0]] == j);

	++i;
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
	int i = 0;
	int right = 0;
#if 0
	float rate = 1;
	net_t *n = net_create(3);

	net_add(n, fc_layer(-28 * 28, 10, 0));
	net_add(n, softmax_layer(10, 10, 0));
	net_add(n, cee_layer(10, 0, -10));
#else
	float rate = 1e-4;
	net_t *n = net_create(12);
	layer_t *dropout = dropout_layer(0, 0);

	net_add(n, conv_layer(1, 28, 28, 32, 28, 28, 5, 1, 0));
	net_add(n, relu_layer(0, 0, 0));
	net_add(n, max_pooling_layer(32, 28, 28, 14, 14, 2, 0, 0));

	net_add(n, conv_layer(32, 14, 14, 64, 14, 14, 5, 1, 0));
	net_add(n, relu_layer(0, 0, 0));
	net_add(n, max_pooling_layer(64, 14, 14, 7, 7, 2, 0, 0));

	net_add(n, fc_layer(0, 1024, 0));
	net_add(n, relu_layer(0, 0, 0));
	net_add(n, dropout);

	net_add(n, fc_layer(0, 10, 0));
	net_add(n, softmax_layer(0, 0, 0));
	net_add(n, cee_layer(0, 0, 0));
#endif
	net_finish(n, TRAIN_ADAM);

	images = mnist_open(argv[1]);
	labels = mnist_open(argv[2]);

	net_param_load(n, "params.bin");

	SET_DROP_PROB(dropout, 0.5);
	for (i = 0; i < 100; ++i)
	{
		int j = 0;
		time_t start = time(NULL);

		net_train(n, feed_data, rate, images->dim[0] / 100);
		LOG("round %d train with rate %f [%ld s]\n", i, rate, time(NULL) - start);

		//LOG("output ");
		//for (j = 0; j <  10; ++j)
		//	LOG("%f[%f] ", LAST_LAYER(n)->in.val[j], LAST_LAYER(n)->param.val[j]);
		//LOG("\n");
		if (i % 10 == 0)
			net_param_save(n, "params.bin");
	}

	net_param_save(n, "params.bin");

	mnist_close(labels);
	mnist_close(images);

	images = mnist_open(argv[3]);
	labels = mnist_open(argv[4]);

	SET_DROP_PROB(dropout, 0);
	for (i = 0; i < images->dim[0]; ++i)
	{
		feed_data(n);
		net_forward(n);

		right += (arg_max(LAST_LAYER(n)->in.val, 10) == arg_max(LAST_LAYER(n)->param.val, 10));
	}

	LOG("accurcy %f\n", 1.0 * right / images->dim[0]);

	mnist_close(labels);
	mnist_close(images);

	net_destroy(n);

	return 0;
}
