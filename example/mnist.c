#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"

#define BIG_ENDIAN_INT32(x) ((((x) & 0xFF) << 24) | ((((x) >> 8) & 0xFF) << 16) | ((((x) >> 16) & 0xFF) << 8) | (((x) >> 24) & 0xFF))

static int isBigEndian()
{
	short int test = 0x1234;
	return (0x12 == *(char *)&test);
}

idx_t* mnist_open(char* file)
{
	idx_t *mnist_data = NULL;
	FILE *fp = fopen(file, "rb");
	if (fp) {
		int i = 0;
		int dataSize = 0;
		int bigEndian = isBigEndian();
		const size_t sizes[0x10] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 4, 4, 8, 0};

		mnist_data = (idx_t *)malloc(sizeof(*mnist_data));
		fread(mnist_data->zero, sizeof(mnist_data->zero), 1, fp);
		fread(&(mnist_data->type), sizeof(mnist_data->type), 1, fp);
		fread(&(mnist_data->dims), sizeof(mnist_data->dims), 1, fp);
		
		//printf("type = %d, dims = %d\n", mnist_data->type, mnist_data->dims);

		mnist_data->dim = (int32_t *)calloc(sizeof(*(mnist_data->dim)), mnist_data->dims);
		for (i = 0; i < mnist_data->dims; ++i) {
			fread(&(mnist_data->dim[i]), sizeof(*(mnist_data->dim)), 1, fp);
		}
		if (!bigEndian) {
			for (i = 0; i < mnist_data->dims; ++i) {
				mnist_data->dim[i] = BIG_ENDIAN_INT32(mnist_data->dim[i]);
			}
		}

		dataSize = sizes[mnist_data->type];
		for (i = 0; i < mnist_data->dims; ++i) {
			//printf("dim[%d] = %d\n", i, mnist_data->dim[i]);
			dataSize *= mnist_data->dim[i];
		}
		
		//printf("dataSize = %d\n", dataSize);
		mnist_data->data = (uint8_t *)malloc(dataSize);
		fread(mnist_data->data, dataSize, 1, fp);
		
		fclose(fp);
	}
	return mnist_data;
}


void mnist_close(idx_t *mnist_data)
{
	if (mnist_data) {
		free(mnist_data->data);
		free(mnist_data->dim);
		free(mnist_data);
	}
}

