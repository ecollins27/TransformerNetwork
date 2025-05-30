#include "MatrixBatch.h"

float MatrixBatch::ALPHA = 1.0f;
float MatrixBatch::BETA = 0.0f;

MatrixBatch::MatrixBatch(int batchSize, int height, int width) {
	maxHeight = height;
	maxWidth = width;
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	cudaError_t err = cudaMalloc(&device, batchSize * sizeof(float*));
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
	deviceArray = new float* [batchSize];
	for (int i = 0; i < batchSize; i++) {
		cudaError_t err = cudaMalloc(&deviceArray[i], maxHeight * maxWidth * sizeof(float));
		if (err != cudaSuccess) {
			throw invalid_argument("CUDA memory allocation failed");
		}
	}
	err = cudaMemcpy(device, deviceArray, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory copy failed");
	}
	host = NULL;
}

int MatrixBatch::e(int i, int j) {
	return i * width + j;
}

float& MatrixBatch::operator()(int b, int i, int j) {
	if (host == NULL) {
		throw invalid_argument("Matrix must be converted to host before accessing");
	}
	return host[b][e(i, j)];
}

void MatrixBatch::allocateHost() {
	host = new float* [batchSize];
	for (int i = 0; i < batchSize; i++) {
		host[i] = new float[maxHeight * maxWidth];
	}
	copyToHost();
}

void MatrixBatch::deallocateHost() {
	copyToDevice();
	for (int i = 0; i < batchSize; i++) {
		delete[] host[i];
	}
	delete[] host;
	host = NULL;
}

void MatrixBatch::copyToDevice() {
	copy(host);
}

void MatrixBatch::copyToHost() {
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(host[i], deviceArray[i], height * width * sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			throw invalid_argument("CUDA memory allocation failed");
		}
	}
}

void MatrixBatch::print() {
	if (host == NULL) {
		throw invalid_argument("Matrix must be converted to host before printing");
	}
	for (int k = 0; k < batchSize; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				printf("%f  ", host[k][e(i, j)]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

void MatrixBatch::setDims(int height, int width) {
	if (height > maxHeight || width > maxWidth) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->height = height;
	this->width = width;
}
void MatrixBatch::copy(float** host_matrix) {
	cudaError_t err;
	for (int i = 0; i < batchSize; i++) {
		err = cudaMemcpy(deviceArray[i], host_matrix[i], height * width * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			throw invalid_argument("CUDA memory allocation failed");
		}
	}
}

void MatrixBatch::multiplyABC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C) {
	cublasStatus_t stat = cublasSgemmBatched(Matrix2::HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, C.width, C.height, A.width, &ALPHA, B.device, C.width, A.device, A.width, &BETA, C.device, C.width, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
}