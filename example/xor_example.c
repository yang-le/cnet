#include "net.h"
#include "log.h"
#include "fc_layer.h"
#include "sigmoid_layer.h"
#include "mse_layer.h"
#include <memory.h>

static void feed_data(net_t *n)
{
	data_val_t data[8] = {0, 0, 0, 1, 1, 0, 1, 1};
	data_val_t label[4] = {0, 1, 1, 0};

	memcpy(n->layer[0]->in.val, data, 8 * sizeof(data_val_t));
	memcpy(LAST_LAYER(n)->extra.val, label, 4 * sizeof(data_val_t));
}

int main(int argc, char **argv)
{
	int i = 0;
	float rate = 10;

	net_t *n = net_create(5, TRAIN_DEFAULT, 4);

	net_add(n, fc_layer(2, 2, FILLER_XAVIER, 0.5, 0));
	net_add(n, sigmoid_layer(2));
	net_add(n, fc_layer(2, 1, FILLER_XAVIER, 0.5, 0));
	net_add(n, sigmoid_layer(1));
	net_add(n, mse_layer(1));

	net_finish(n);

	for (i = 0; i < 3000; ++i)
	{
		//LOG("train with rate %f\n", rate);
		net_train(n, feed_data, rate);
	}

	feed_data(n);
	net_forward(n);
	for (i = 0; i < 4; ++i)
	{
		LOG("input %f %f, output %f\n", n->layer[0]->in.val[2 * i], n->layer[0]->in.val[2 * i + 1], LAST_LAYER(n)->in.val[i]);
	}

	net_destroy(n);

	return 0;
}
