#include "layers.h"

int main(int argc, char **argv)
{
    net_t *n, *n1, *n2;

    NET_CREATE(n1, TRAIN_SGD, 1);
    NET_ADD(n1, relu_layer(1));
    NET_FINISH(n1);

    NET_CREATE(n2, TRAIN_SGD, 1);
    NET_ADD(n2, relu_layer(1));
    NET_FINISH(n2);

    NET_CREATE(n, TRAIN_SGD, 1);
    NET_ADD(n, branch_layer(2, n1, 0, n2, 1, NULL));
    NET_ADD(n, merge_layer(2, n1, 1, n2, 0, NULL));
    NET_FINISH(n);

    n->layer[0]->in.val[0] = 1;
    n->layer[0]->in.val[1] = 2;

    net_forward(n);

    n->layer[1]->out.grad[0] = 1;
    n->layer[1]->out.grad[1] = 2;

    net_backward(n);

    return 0;
}
