#include "net.h"
#include "log.h"
#include "fc_layer.h"
#include "sigmoid_layer.h"
#include "mse_layer.h"

static void feed_data(net_t *n)
{
	static unsigned int i = 0;
	
	data_val_t data0[4] = {0, 0, 1, 1};
	data_val_t data1[4] = {0, 1, 0, 1};
	data_val_t label[4] = {0, 1, 1, 0};

	n->layer[0]->in.val[0] = data0[i % 4];
	n->layer[0]->in.val[1] = data1[i % 4];
	LAST_LAYER(n)->param.val[0] = label[i % 4];

	++i;
}

int main(int argc, char** argv)
{
	int i = 0;
	float rate = 10;

	net_t *n = net_create(5);

	net_add(n, fc_layer(-2, 2, 0));
	net_add(n, sigmoid_layer(0, 2, 0));
	net_add(n, fc_layer(0, 1, 0));
	net_add(n, sigmoid_layer(0, 1, 0));
	net_add(n, mse_layer(0, 1, -1));
	
	net_finish(n);

	for (i = 0; i < 3000; ++i)
	{
		//LOG("train with rate %f\n", rate);
		net_train(n, feed_data, rate, 4);
	}

	for (i = 0; i < 4; ++i)
	{
		feed_data(n);
		net_forward(n);
		LOG("input %f %f, output %f\n", n->layer[0]->in.val[0], n->layer[0]->in.val[1], LAST_LAYER(n)->in.val[0]);
	}

	net_destroy(n);

	return 0;
}
