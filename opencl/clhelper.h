#pragma once

#include <CL/opencl.h>

#ifdef __cplusplus
extern "C" {
#endif

cl_int cl_init(void);
void cl_deinit(void);
cl_context cl_get_context(int plat);
cl_command_queue cl_get_queues(int plat, int dev);

#ifdef __cplusplus
}
#endif
