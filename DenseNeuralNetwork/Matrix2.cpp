#include "Matrix2.h"

Matrix2::Matrix2(int height, int width) {
	maxHeight = height;
	maxWidth = width;
	this->height = height;
	this->width = width;
	heightRange = new int[maxHeight];
	widthRange = new int[maxWidth];
	matrix = new float[maxHeight * maxWidth];
	for (int i = 0; i < height; i++) {
		heightRange[i] = i;
		for (int j = 0; j < width; j++) {
			if (i == 0) {
				widthRange[j] = j;
			}
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

void Matrix2::dotProduct::operator()(int j) {
	float sum = 0;
	for (int k = 0; k < A.width; k++) {
		sum += A(i, k) * B(k, j);
	}
	C(i, j) = sum;
}

void Matrix2::rowMultiply::operator()(int i) {
	thrust::for_each_n(thrust::host, C.widthRange, C.width, dotProduct(A, B, C, i));
}

void Matrix2::multiplyABC(Matrix2& A, Matrix2& B, Matrix2& C) {
	if (A.width != B.height || C.height != A.height || C.width != B.width) {
		throw invalid_argument("Incompatible matrices");
	}
	thrust::for_each_n(thrust::host, C.heightRange, C.height, rowMultiply(A, B, C));
}