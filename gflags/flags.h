#pragma once

#ifdef __cplusplus
#include <gflags/gflags.h>

DECLARE_string(load_weights);
DECLARE_string(save_weights);
DECLARE_int32(cl_platform);
DECLARE_int32(cl_device);
#endif

#ifdef __cplusplus
extern "C" {
#endif

void gflags_parse(int *argc, char ***argv);

#ifdef __cplusplus
}
#endif
