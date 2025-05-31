#pragma once
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

using namespace std;

class Matrix2 {

public:
	float* device;
	float* host = NULL;
	int maxHeight, maxWidth;
	int height, width;

	static float ALPHA;
	static float BETA;
	static int THREADS_PER_BLOCK;
	static cublasHandle_t HANDLE;

	Matrix2() {};
	Matrix2(int height, int width);
	int e(int i, int j);
	float& operator()(int i, int j);
	void print();
	void allocateHost();
	void deallocateHost();
	void copyToDevice();
	void copy(float* matrix);
	void setDims(int height, int width);

	__global__
	void kernelAdd(int N, const float* A, const float* B, const float* C);
	__global__
	void kernelMultiply(int N, const float* A, const float* B, const float* C);

	static void multiplyABC(Matrix2& A, Matrix2& B, Matrix2& C);
	static void add(Matrix2& A, Matrix2& B, Matrix2& C);
	static void elementMultiply(Matrix2& A, Matrix2& B, Matrix2& C);
};