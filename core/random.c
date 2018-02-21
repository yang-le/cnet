#include "random.h"

void uniform(float data[], size_t size, float a, float b)
{
	if (a > b)
	{
		float c = a;
		a = b;
		b = c;
	}

	for (unsigned int i = 0; i < size; ++i)
		data[i] = a + (b - a) * rand() / RAND_MAX;
}

void normal(float data[], size_t size, float mean, float std_dev)
{
	float u, v, w;

	for (unsigned int i = 0; i < size; ++i)
	{
		do {
			u = (float)(2. * rand() / RAND_MAX - 1.);
			v = (float)(2. * rand() / RAND_MAX - 1.);
			w = u * u + v * v;
		} while ((w < 0) || (w > 1));

		// see Box-Muller
		data[i] = std_dev * (float)(u * sqrt(-2 * log(w) / w)) + mean;
	}
}
