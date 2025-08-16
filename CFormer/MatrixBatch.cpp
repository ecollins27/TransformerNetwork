// Compile with CUDA

#include "MatrixBatch.h"
#include "Matrix.h"

MatrixBatch::MatrixBatch(int batchSize, int height, int width) {
	maxLength = length;
	this->length = maxLength;
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	cudaError_t err = cudaMallocHost(&host, batchSize * sizeof(float*));
	for (int i = 0; i < batchSize; i++) {
		err = cudaMallocHost(&host[i], maxLength * sizeof(float));
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
		}
	}
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < length; j++) {
			host[i][j] = 0;
		}
	}
}

MatrixBatch::MatrixBatch(FillFunction& fillFunction, int batchSize, int height, int width) {
	maxLength = height * width;
	this->length = maxLength;
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	cudaError_t err = cudaMallocHost(&host, batchSize * sizeof(float*));
	for (int i = 0; i < batchSize; i++) {
		err = cudaMallocHost(&host[i], maxLength * sizeof(float));
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
		}
	}
	fill(fillFunction);
}

void MatrixBatch::free() {
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaFreeHost(host[i]);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
		}
	}
	err = cudaFreeHost(host);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
	}
}

int MatrixBatch::e(int i, int j) {
	return i + height * j;
}

float& MatrixBatch::operator()(int i, int j, int k) {
	return host[i][e(j, k)];
}

void MatrixBatch::fill(FillFunction& fillFunction) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < width; k++) {
				host[i][e(j, k)] = fillFunction(j, k);
			}
		}
	}
}

void MatrixBatch::print() {
	for (int n = 0; n < batchSize; n++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				printf("%f  ", host[n][e(i, j)]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

void MatrixBatch::setDims(int height, int width) {
	if (height * width > maxLength) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->height = height;
	this->width = width;
	this->length = height * width;
}

void MatrixBatch::setHeight(int height) {
	if (height * width > maxLength) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->height = height;
	this->length = height * width;
}

void MatrixBatch::setWidth(int width) {
	if (height * width > maxLength) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->width = width;
	this->length = height * width;
}

MatrixBatch* MatrixBatch::allocateMatrixBatchArray(FillFunction& fill, int arrayLength, int batchSize, int height, int width) {
	MatrixBatch* array = new MatrixBatch[arrayLength];
	for (int i = 0; i < arrayLength; i++) {
		array[i] = MatrixBatch(fill, batchSize, height, width);
	}
	return array;
}

MatrixBatch* MatrixBatch::allocateMatrixBatchArray(int arrayLength, int batchSize, int height, int width) {
	MatrixBatch* array = new MatrixBatch[arrayLength];
	for (int i = 0; i < arrayLength; i++) {
		array[i] = MatrixBatch(batchSize, height, width);
	}
	return array;
}
