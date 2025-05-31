﻿#include "Matrix2.h"

float Matrix2::ALPHA = 1.0f;
float Matrix2::BETA = 0.0f;
int Matrix2::THREADS_PER_BLOCK = 256;
cublasHandle_t Matrix2::HANDLE = NULL;

Matrix2::Matrix2(int height, int width) {
	maxHeight = height;
	maxWidth = width;
	this->height = height;
	this->width = width;
	cudaError_t err = cudaMalloc(&device, maxHeight * maxWidth * sizeof(float));
	if (err != cudaSuccess) {
		throw invalid_argument("CUDA memory allocation failed");
	}
}

int Matrix2::e(int i, int j) {
	return i * width + j;
}

float& Matrix2::operator()(int i, int j) {
	if (host == NULL) {
		throw invalid_argument("Matrix must be converted to host before accessing");
	}
	return host[e(i, j)];
}

void Matrix2::copy(float* host_matrix) {
	cudaMemcpy(device, host_matrix, height * width * sizeof(float), cudaMemcpyHostToDevice);
}

void Matrix2::allocateHost() {
	host = new float[maxHeight * maxWidth];
	cudaMemcpy(host, device, height * width * sizeof(float), cudaMemcpyDeviceToHost);
}

void Matrix2::deallocateHost() {
	cudaMemcpy(device, host, height * width * sizeof(float), cudaMemcpyHostToDevice);
	delete[] host;
	host = NULL;
}

void Matrix2::copyToDevice() {
	copy(host);
}

void Matrix2::print() {
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

void Matrix2::setDims(int height, int width) {
	if (height > maxHeight || width > maxWidth) {
		throw invalid_argument("Set dimensions exceed max dimensions");
	}
	this->height = height;
	this->width = width;
}

__global__
void Matrix2::kernelAdd(int N, const float* A, const float* B, const float* C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = A[i] + B[i];
	}
}

__global__
void Matrix2::kernelMultiply(int N, const float* A, const float* B, const float* C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = A[i] * B[i];
	}
}

void Matrix2::multiplyABC(Matrix2& A, Matrix2& B, Matrix2& C) {
	cublasStatus_t stat = cublasSgemm(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, C.width, C.height, A.width, &ALPHA, B.device, C.width, A.device, A.width, &BETA, C.device, C.width);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
}

void Matrix2::elementMultiply(Matrix2& A, Matrix2& B, Matrix2& C) {
	int N = A.height * A.width;
	int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	kernelMultiply  <<< numBlocks, THREADS_PER_BLOCK >>> (N, A.device, B.device, C.device);
}

void Matrix2::kernalAdd(Matrix2& A, Matrix2& B, Matrix2& C) {
	int N = A.height * A.width;
	int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	kernelAdd <<< numBlocks, THREADS_PER_BLOCK >>> (N, A.device, B.device, C.device);
}