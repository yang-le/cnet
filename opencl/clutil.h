#ifndef __CL_UTIL_H__
#define __CL_UTIL_H__

#include <CL/opencl.h>

#ifdef __cplusplus
extern "C" {
#endif

cl_int getPlatforms(cl_platform_id **platforms, cl_uint *num);
char* getPlatformName(cl_platform_id platform);
cl_int getDevices(cl_platform_id platform, cl_device_type type, cl_device_id **devices, cl_uint *num);
char* getDeviceName(cl_device_id device);
int isPlatformHaveGPU(cl_platform_id platform);
int getPlatformDeviceNum(cl_platform_id platform);
cl_uint getNDRange(cl_device_id device, size_t **ranges);
cl_int loadProgramFromFile(cl_context context, char *filename, cl_program *program);
char* getProgramBuildLog(cl_program program, cl_device_id device);

#ifdef __cplusplus
}
#endif

#endif // __CL_UTIL_H__

