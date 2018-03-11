#include "layers.h"

int main(int argc, char **argv)
{
    branch_layer(2, relu_layer(1), 0, relu_layer(1), 1, NULL);
    return 0;
}
