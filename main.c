#include "vld.h"
#include "net.h"
#include "conv_layer.h"
#include "pooling_layer.h"
#include "relu_layer.h"
#include "fc_layer.h"
#include "softmax_layer.h"
#include "cee_layer.h"

#if 0
CNET_INLINE void neg_forward(data_t *in, data_t *out)
{
	out->val = -in->val;	
}

CNET_INLINE void neg_backward(data_t *in, data_t *out)
{
	out->grad += -in->grad;
}

CNET_INLINE void inv_forward(data_t *in, data_t *out)
{
	out->val = 1 / in->val;	
}

CNET_INLINE void inv_backward(data_t *in, data_t *out)
{
	out->grad += -in->grad / (out->val * out->val);
}

CNET_INLINE void plus_forward(data_t *in0, data_t *in1, data_t *out)
{
	out->val = in0->val + in1->val;
}

CNET_INLINE void plus_backward(data_t *in, data_t *out)
{
	out->grad += in->grad;
}

CNET_INLINE void mul_forward(data_t *in0, data_t *in1, data_t *out)
{
	out->val = in0->val * in1->val;
}

CNET_INLINE void mul_backward(data_t *in, data_t *out)
{
	out->grad += in->grad * in->val / out->val;
}
#endif

int main(int argc, char **argv)
{
	linear_sample();
	return 0;
}
