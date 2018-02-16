#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>

#define UBYTE	0x08
#define BYTE	0x09
#define SHORT	0x0B	// 2 bytes
#define INT	0x0C	// 4 bytes
#define FLOAT	0x0D	// 4 bytes
#define DOUBLE	0x0E	// 8 bytes

typedef struct {
	uint8_t zero[2];
	uint8_t type;
	uint8_t dims;
	int32_t *dim;
	uint8_t *data;
} idx_t;

#ifdef __cplusplus
extern "C" {
#endif

idx_t* mnist_open(char* file);
void mnist_close(idx_t *mnist_data);

#ifdef __cplusplus
}
#endif

#endif // MNIST_H

