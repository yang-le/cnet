#include "vgg.h"

int main(int argc, char **argv)
{
    net_t *n = NULL;

    NET_CREATE_VGG_E(n, TRAIN_SGD, 16);

    return 0;
}
