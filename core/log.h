#pragma once

#include <stdio.h>

#if 0
#define LOG(fmt, ...)
#else
#define LOG(fmt, ...) printf(fmt, ##__VA_ARGS__)
#endif
