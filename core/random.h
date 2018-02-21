#pragma once

#include <stdlib.h>
#include <math.h>

void uniform(float data[], size_t size, float a, float b);
void normal(float data[], size_t size, float mean, float std_dev);

#define truncated_normal(x, mean, std_dev) do { \
    normal((x), 1, (mean), (std_dev)); \
} while(fabs(*(x) - (mean)) > 2 * (std_dev))
