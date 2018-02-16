#include "net.h"
#include "log.h"
#include "fc_layer.h"
#include "mse_layer.h"

static void feed_data(net_t *n)
{
	static int i = 0;
	
	data_val_t data[4] = {0, 1, 2, 3};
	data_val_t label[4] = {6, 14, 26, 34};

	n->layer[0]->in.val[0] = data[i % 4];
	LAST_LAYER(n)->param.val[0] = label[i % 4];

	++i;
}

void linear_sample()
{
	int i = 0;
	float rate = 1e-2;

	net_t *n = net_create(2);

	net_add(n, fc_layer(-1, 1, 0));
	net_add(n, mse_layer(1, 1, -1));
	
	net_finish(n);

	for (i = 0; i < 20; ++i)
	{
		LOG("train with rate %f\n", rate);
		net_train(n, feed_data, rate, 100);
		//rate *= 2;
	}

	net_destroy(n);
}
