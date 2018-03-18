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

static idx_t *images = NULL;
static idx_t *labels = NULL;

static void feed_data(net_t *n)
{
    for (int b = 0; b < n->batch; ++b)
    {
        int j = 0;
        int i = rand();
        for (j = 0; j < 28 * 28; ++j)
            n->layer[0]->in.val[b * 28 * 28 + j] = images->data[(i % images->dim[0]) * 28 * 28 + j] / 255.0 - 0.5;

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

    NET_CREATE(n, TRAIN_ADAGRAD, 50);

    NET_ADD(n, rnn_layer(28, 10, 50, 28, FILLER_XAVIER, 0.5, 0));
    NET_ADD(n, fc_layer(0, 10, FILLER_XAVIER, 0.5, 0));
    NET_ADD(n, softmax_layer(0));
    NET_ADD(n, cee_layer(0));

    NET_FINISH(n);

    images = mnist_open(argv[1]);
    labels = mnist_open(argv[2]);

    net_param_load(n, "params.bin");

    for (int i = 0; i < 20000; ++i)
    {
        net_train(n, feed_data, 1e-1);

        for (int b = 0; b < n->batch; ++b)
            loss += LAST_LAYER(n)->out.val[b];

        if (i % 500 == 0)
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
#endif
            for (int b = 0; b < n->batch; ++b)
                right += (arg_max(&LAST_LAYER(n)->in.val[b * 10], 10) == arg_max(&LAST_LAYER(n)->extra.val[b * 10], 10));

            LOG("loss = %f, train accurcy = %f, step = %d\n", loss / n->batch / 500, 1.0 * right / n->batch, i);
            LOG("steps/sec = %f\n", 500.0 / (time(NULL) - start));

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
