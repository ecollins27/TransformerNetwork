// Compile with CUDA

#include "Matrix.h"

Matrix::Matrix(int height, int width) {
	maxLength = height * width;
	this->length = maxLength;
	this->height = height;
	this->width = width;
	cudaError_t err = cudaMallocHost(&host, maxLength * sizeof(float));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	for (int i = 0; i < maxLength; i++) {
		host[i] = 0;
	}
}

Matrix::Matrix(FillFunction& fillFunction, int height, int width) {
	maxLength = height * width;
	this->length = maxLength;
	this->height = height;
	this->width = width;
	cudaError_t err = cudaMallocHost(&host, maxLength * sizeof(float));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	fill(fillFunction);
}

void Matrix::free() {
	cudaError_t err = cudaFreeHost(host);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
	}
}

int Matrix::e(int i, int j) {
	return i + height * j;
}

float& Matrix::operator()(int i, int j) {
	return host[e(i, j)];
}

void Matrix::fill(FillFunction& fillFunction) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			host[e(i, j)] = fillFunction(i, j);
		}
	}
}

void Matrix::print() {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%f  ", host[e(i, j)]);
		}
		printf("\n");
	}
}

void Matrix::setDims(int height, int width) {
	if ((isLayerOutput && height * (width + 1) > maxLength) || (!isLayerOutput && height * width > maxLength)) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->height = height;
	this->width = width;
	this->length = height * width;
}

void Matrix::setHeight(int height) {
	if ((isLayerOutput && height * (width + 1) > maxLength) || (!isLayerOutput && height * width > maxLength)) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->height = height;
	this->length = height * width;
}

void Matrix::setWidth(int width) {
	if ((isLayerOutput && height * (width + 1) > maxLength) || (!isLayerOutput && height * width > maxLength)) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->width = width;
	this->length = height * width;
}

void Matrix::setLayerOutput(bool isLayerOutput) {
	if (this->isLayerOutput == isLayerOutput) {
		return;
	}
	else if (this->isLayerOutput && !isLayerOutput) {
		width++;
		length = height * width;
		this->isLayerOutput = isLayerOutput;
	}
	else {
		width--;
		length = height * width;
		this->isLayerOutput = isLayerOutput;
	}
}

MatrixBatch Matrix::subMatrixBatch(int numMatrices, int subWidth) {
	MatrixBatch matrixBatch;
	matrixBatch.batchSize = numMatrices;
	matrixBatch.height = height;
	matrixBatch.width = subWidth;
	matrixBatch.maxLength = height * subWidth;
	matrixBatch.length = matrixBatch.maxLength;
	cudaError_t err = cudaMallocHost(&matrixBatch.host, numMatrices * sizeof(float*));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	for (int i = 0; i < numMatrices; i++) {
		matrixBatch.host[i] = &host[i * subWidth * height];
	}
	return matrixBatch;
}

Matrix* Matrix::allocateMatrixArray(FillFunction& fillFunction, int batchSize, int height, int width) {
	Matrix* array = new Matrix[batchSize];
	for (int i = 0; i < batchSize; i++) {
		array[i] = Matrix(fillFunction, height, width);
	}
	return array;
}

Matrix* Matrix::allocateMatrixArray(int batchSize, int height, int width) {
	Matrix* array = new Matrix[batchSize];
	for (int i = 0; i < batchSize; i++) {
		array[i] = Matrix(height, width);
	}
	return array;
}