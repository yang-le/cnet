#pragma once

#include <CL/opencl.h>

#ifdef __cplusplus
extern "C" {
#endif

cl_int cl_init(void);
void cl_deinit(void);
cl_context cl_get_context(int plat);
cl_command_queue cl_get_queues(int plat, int dev);

void cl_set_default_plat(int plat);
void cl_set_default_dev(int dev);
cl_context cl_get_default_context(void);
cl_command_queue cl_get_default_queues(void);

#ifdef __cplusplus
}
#endif
