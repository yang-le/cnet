#include <stdio.h>
#include "clutil.h"

static cl_uint num_platforms = 0;
static cl_platform_id *platforms = NULL;
static cl_uint *num_devices = 0;
static cl_device_id **devices = NULL;
static cl_context *contexts = NULL;

void cl_init()
{
    cl_int clRet = 0;

    clRet = getPlatforms(&platforms, &num_platforms);
    if (clRet != CL_SUCCESS)
    {
        printf("getPlatforms failed, err is %d\n", clRet);
        return clRet;
    }

    printf("got %d platform(s)\n", num_platforms);

    num_devices = (cl_uint *)calloc(num_platforms, sizeof(cl_uint));
    if (NULL == num_devices)
    {
        printf("alloc mem for num_devices failed!\n");
        return CL_OUT_OF_HOST_MEMORY;
    }

    devices = (cl_device_id **)calloc(num_platforms, sizeof(cl_device_id *));
    if (NULL == devices)
    {
        printf("alloc mem for devices failed!\n");
        return CL_OUT_OF_HOST_MEMORY;
    }

    contexts = (cl_context *)calloc(num_platforms, sizeof(cl_context));
    if (NULL == contexts)
    {
        printf("alloc mem for contexts failed!\n");
        return CL_OUT_OF_HOST_MEMORY;
    }

    for (int i = 0; i < num_platforms; ++i)
    {
        char *name = getPlatformName(platforms[i]);
        if (name)
        {
            printf("platform[%d] is %s\n", i, name);
        }

        free(name);

        clRet = getDevices(platforms[i], CL_DEVICE_TYPE_ALL, &(devices[i]), &num_devices[i]);
        if (clRet != CL_SUCCESS)
        {
            printf("getDevices failed, err is %d\n", clRet);
            continue;
        }

        for (int j = 0; j < num_devices[i]; ++j)
        {
            char *name = getDeviceName(devices[i][j]);
            if (name)
            {
                printf("\t device[%d] is %s\n", j, name);
            }

            free(name);

            size_t *ranges = NULL;
            cl_uint dims = getNDRange(devices[i][j], &ranges);
            printf("\t NDRange = ");
            for (int k = 0; k < dims; ++k)
                printf("%ld ", ranges[k]);
            printf("\n");

            free(ranges);
        }

        contexts[i] = clCreateContext(NULL, num_devices[i], devices[i], NULL, NULL, &clRet);
        if (clRet != CL_SUCCESS)
        {
            printf("clCreateContextFromType failed, ret = %d\n", clRet);
        }
    }
}

void cl_deinit()
{
    free(contexts);
    free(num_devices);
    free(devices);
    free(platforms);
}

cl_context cl_get_context(int id)
{
    return contexts[id];
}
