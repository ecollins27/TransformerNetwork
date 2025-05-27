#include "Matrix2.h"

Matrix2::Matrix2(int height, int width) {
	maxHeight = height;
	maxWidth = width;
	this->height = height;
	this->width = width;
	if (ON_DEVICE) {
		cudaMallocManaged(&matrix, maxHeight * maxWidth * sizeof(float));
	}
	else {
		matrix = new float[maxHeight * maxWidth];
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			matrix[e(i, j)] = 0;
		}
	}
}

int Matrix2::e(int i, int j) {
	return i * width + j;
}

float& Matrix2::operator()(int i, int j) {
	return matrix[e(i, j)];
}

void Matrix2::print() {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%f  ", matrix[e(i, j)]);
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

void Matrix2::multiplyABC(Matrix2& A, Matrix2& B, Matrix2& C) {
	cublasHandle_t handle;
	cublasCreate(&handle);

	float alpha = 1.0f;
	float beta = 0.0f;
	int M = C.height;
	int N = C.width;
	int K = A.width;
	cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, C.width, C.height, A.width, &alpha, B.matrix, C.width, A.matrix, A.width, &beta, C.matrix, N);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("cuBLAS multiplication failed");
	}
}