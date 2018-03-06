#include "net.h"
#include "log.h"
#include "fc_layer.h"
#include "mse_layer.h"
#include <memory.h>

static void feed_data(net_t *n)
{
	data_val_t data[4] = {0, 1, 2, 3};
	data_val_t label[4] = {5, 15, 25, 35};

	memcpy(n->layer[0]->in.val, data, 4 * sizeof(data_val_t));
	memcpy(LAST_LAYER(n)->extra.val, label, 4 * sizeof(data_val_t));
}

int main(int argc, char **argv)
{
	int i = 0;
	float rate = 1e-1;

	net_t *n = net_create(2, TRAIN_DEFAULT, 4);

	net_add(n, fc_layer(1, 1, FILLER_CONST, 0, 0));
	net_add(n, mse_layer(1));

	net_finish(n);

	for (i = 0; i < 100; ++i)
	{
		//LOG("train with rate %f\n", rate);
		net_train(n, feed_data, rate);
		//rate *= 2;
	}

	LOG("result = %f %f\n", n->layer[0]->weight.val[0], n->layer[0]->bias.val[0]);

	net_destroy(n);

	return 0;
}
