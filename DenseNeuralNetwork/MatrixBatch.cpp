#include "MatrixBatch.h"

float MatrixBatch::ALPHA;
float MatrixBatch::BETA;

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
	for (int i = 0; i < batchSize; i++) {
		cudaError_t err = cudaMalloc(&device[i], maxHeight * maxWidth * sizeof(float));
		if (err != cudaSuccess) {
			throw invalid_argument("CUDA memory allocation failed");
		}
	}
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
	cudaMemcpy2D(host, batchSize, device, batchSize, batchSize, height * width * sizeof(float), cudaMemcpyDeviceToHost);
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

void MatrixBatch::print() {
	if (host == NULL) {
		throw invalid_argument("Matrix must be converted to host before printing");
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%f  ", host[e(i, j)]);
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
	cudaMemcpy2D(device, batchSize, host_matrix, batchSize, batchSize, height * width * sizeof(float), cudaMemcpyHostToDevice);
}

void MatrixBatch::multiplyABC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C) {
	cublasStatus_t stat = cublasSgemmBatched(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, C.width, C.height, A.width, &ALPHA, B.device, C.width, A.device, A.width, &BETA, C.device, C.width, C.batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
}