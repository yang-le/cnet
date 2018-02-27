#include <stdio.h>
#include "clutil.h"

char* getProgramBuildLog(cl_program program, cl_device_id device)
{
	cl_int clRet = 0;
	size_t len = 0;
	char *name = NULL;

	clRet = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
	if ((0 == len) || (clRet != CL_SUCCESS)) {
		return NULL;
	}

	name = (char *)malloc(len);
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, name, NULL);
	return name;
}

cl_int loadProgramFromFile(cl_context context, char *filename, cl_program *program)
{
	cl_int clRet = 0;

	if ((NULL == filename) || (NULL == program)) {
		return CL_INVALID_VALUE;
	}

	FILE *fp = fopen(filename, "r");
	if (NULL == fp) {
		return CL_INVALID_VALUE;
	}

	fseek(fp, 0L, SEEK_END);
	long filelen =ftell(fp);

	char *source = (char *)malloc(filelen + 1);
	if (NULL == source) {
		return CL_OUT_OF_HOST_MEMORY;
	}

	fseek(fp, 0L, SEEK_SET);
	fread(source, filelen, 1, fp);
	fclose(fp);

	source[filelen] = '\0';
	*program = clCreateProgramWithSource(context, 1, (const char**)(&source), NULL, &clRet);

	return clRet;
}

cl_uint getNDRange(cl_device_id device, size_t **ranges)
{
	cl_int clRet = 0;
	cl_uint dims;

	clRet = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &dims, NULL);
	if ((0 == dims) || (clRet != CL_SUCCESS)) {
		return 0;
	}

	if (ranges) {
		*ranges = (size_t *)calloc(dims, sizeof(size_t));
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, dims * sizeof(size_t), *ranges, NULL);
	}

	return dims;
}

int isPlatformHaveGPU(cl_platform_id platform)
{
	cl_int clRet = 0;
	cl_uint num = 0;

	clRet = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num);
	if (clRet != CL_SUCCESS) {
		return 0;
	}

	return num;
}

int getPlatformDeviceNum(cl_platform_id platform)
{
	cl_int clRet = 0;
	cl_uint num = 0;

	clRet = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num);
	if (clRet != CL_SUCCESS) {
		return 0;
	}

	return num;
}

cl_int getPlatforms(cl_platform_id **platforms, cl_uint *num)
{
	cl_int clRet = 0;

	if ((NULL == platforms) || (NULL == num)) {
		return CL_INVALID_VALUE;
	}

	clRet = clGetPlatformIDs(0, NULL, num);
	if (clRet != CL_SUCCESS) {
		return clRet;
	}

	if (0 == *num) {
		*platforms = NULL;
		return CL_SUCCESS;
	}

	*platforms = (cl_platform_id *)calloc(*num, sizeof(cl_platform_id));
	if (NULL == *platforms) {
		return CL_OUT_OF_HOST_MEMORY;
	}

	return clGetPlatformIDs(*num, *platforms, NULL);
}

char* getPlatformName(cl_platform_id platform)
{
	cl_int clRet = 0;
	size_t len = 0;
	char *name = NULL;

	clRet = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &len);
	if ((0 == len) || (clRet != CL_SUCCESS)) {
		return NULL;
	}

	name = (char *)malloc(len);
	clGetPlatformInfo(platform, CL_PLATFORM_NAME, len, name, NULL);
	return name;
}

cl_int getDevices(cl_platform_id platform, cl_device_type type, cl_device_id **devices, cl_uint *num)
{
	cl_int clRet = 0;

	if ((NULL == devices) || (NULL == num)) {
		return CL_INVALID_VALUE;
	}	

	clRet = clGetDeviceIDs(platform, type, 0, NULL, num);
	if (clRet != CL_SUCCESS) {
		return clRet;
	}

	if (0 == *num) {
		*devices = NULL;
		return CL_SUCCESS;
	}

	*devices = (cl_device_id *)calloc(*num, sizeof(cl_device_id));
	if (NULL == *devices) {
		return CL_OUT_OF_HOST_MEMORY;
	}

	return clGetDeviceIDs(platform, type, *num, *devices, NULL);
}

char* getDeviceName(cl_device_id device)
{
	cl_int clRet = 0;
	size_t len = 0;
	char *name = NULL;

	clRet = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &len);
	if ((0 == len) || (clRet != CL_SUCCESS)) {
		return NULL;
	}

	name = (char *)malloc(len);
	clGetDeviceInfo(device, CL_DEVICE_NAME, len, name, NULL);
	return name;
}

