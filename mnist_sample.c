#include "mnist.h"
#include "net.h"
#include "log.h"
#include "fc_layer.h"
#include "softmax_layer.h"
#include "cee_layer.h"

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

int arg_max(data_val_t *data, int n)
{
	int i = 0;
	int max = 0;
	
	for (i = 0; i < n; ++i)
		if (data[i] > data[max])
			max = i;

	return max;
}

#if 0
	net_t *n = net_create(11);

	net_add(n, conv_layer(1, 28, 28, 32, 28, 28, 5, 1, 0));
	net_add(n, relu_layer(0, 0, 0));
	net_add(n, max_pooling_layer(32, 28, 28, 14, 14, 0, 0, 0));

	net_add(n, conv_layer(32, 14, 14, 64, 14, 14, 5, 1, 0));
	net_add(n, relu_layer(0, 0, 0));
	net_add(n, max_pooling_layer(64, 14, 14, 7, 7, 0, 0, 0));

	net_add(n, fc_layer(0, 1024, 0));
	net_add(n, relu_layer(0, 0, 0));

	net_add(n, fc_layer(0, 10, 0));
	net_add(n, softmax_layer(0, 0, 0));
	net_add(n, cee_layer(0, 0, 0));

	net_finish(n);

	net_destroy(n);
#endif

void mnist_sample(void)
{
	int i = 0;
	int right = 0;
	float rate = 1e-3;

	net_t *n = net_create(3);

	net_add(n, fc_layer(-28 * 28, 10, 0));
	net_add(n, softmax_layer(10, 10, 0));
	net_add(n, cee_layer(10, 0, -10));
	
	net_finish(n);

	images = mnist_open("F:/temp/cnet/cnet/mnist/train-images-idx3-ubyte");
	labels = mnist_open("F:/temp/cnet/cnet/mnist/train-labels-idx1-ubyte");

	for (i = 0; i < 20; ++i)
	{
		int j = 0;

		LOG("round %d train with rate %f\n", i, rate);
		net_train(n, feed_data, rate, 60000);

		feed_data(n);
		net_forward(n);

		LOG("output ");
		for (j = 0; j <  10; ++j)
			LOG("%f[%f] ", LAST_LAYER(n)->in.val[j], LAST_LAYER(n)->param.val[j]);
		LOG("\n");
	}

	mnist_close(labels);
	mnist_close(images);

	images = mnist_open("F:/temp/cnet/cnet/mnist/t10k-images-idx3-ubyte");
	labels = mnist_open("F:/temp/cnet/cnet/mnist/t10k-labels-idx1-ubyte");

	for (i = 0; i < 10000; ++i)
	{
		feed_data(n);
		net_forward(n);

		right += (arg_max(LAST_LAYER(n)->in.val, 10) == arg_max(LAST_LAYER(n)->param.val, 10));
	}

	LOG("accurcy %f\n", 1.0 * right / 10000);

	mnist_close(labels);
	mnist_close(images);

	net_destroy(n);
}
