#pragma once

#include <CL/opencl.h>

void cl_init();
void cl_deinit();
cl_context cl_get_context(int id);
