#include <stdio.h>
#include "clutil.h"

static cl_uint num_platforms = 0;
static cl_platform_id *platforms = NULL;
static cl_uint *num_devices = 0;
static cl_device_id **devices = NULL;
static cl_context *contexts = NULL;
static cl_command_queue **queues = NULL;

cl_int cl_init(void)
{
    cl_int clRet = 0;

    clRet = getPlatformIDs(&platforms, &num_platforms);
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

    queues = (cl_command_queue **)calloc(num_platforms, sizeof(cl_command_queue *));
    if (NULL == queues)
    {
        printf("alloc mem for queues failed!\n");
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

        queues[i] = (cl_command_queue *)calloc(num_devices[i], sizeof(cl_command_queue));
        if (NULL == queues[i])
        {
            printf("alloc mem for queues failed!\n");
            return CL_OUT_OF_HOST_MEMORY;
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

        for (int j = 0; j < num_devices[i]; ++j)
        {
            queues[i][j] = clCreateCommandQueue(contexts[i], devices[i][j], 0, &clRet);
            if (clRet != CL_SUCCESS)
            {
                printf("clCreateCommandQueue failed, ret = %d\n", clRet);
            }
        }
    }

    return clRet;
}

void cl_deinit(void)
{
    free(contexts);
    free(num_devices);
    for (int i = 0; i < num_platforms; ++i)
    {
        free(devices[i]);
        free(queues[i]);
    }
    free(devices);
    free(queues);
    free(platforms);
}

cl_context cl_get_context(int plat)
{
    return contexts[plat];
}

cl_command_queue cl_get_queues(int plat, int dev)
{
    return queues[plat][dev];
}
