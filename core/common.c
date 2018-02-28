#include "common.h"

static size_t alloc_size = 0;

void* alloc(size_t n, size_t size)
{
    void *buf = calloc(n, size);
    
    if (buf)
        alloc_size += n * size;
    
    return buf;
}

size_t get_alloc_size(void)
{
    return alloc_size;
}
