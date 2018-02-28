#pragma once

#include <stdlib.h>

#ifdef _MSC_VER
#define CNET_INLINE __inline
#else
#define CNET_INLINE inline
#endif

void* alloc(size_t n, size_t size);
size_t get_alloc_size(void);
