#pragma once
#include "MatrixBatch.h"
#include "FillFunction.h"
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <xmmintrin.h>

using namespace std;

class Matrix2 {

public:

	static float ALPHA;
	static float BETA0, BETA1;
	static int THREADS_PER_BLOCK;
	static cublasHandle_t HANDLE;


	float* device;
	float* host = NULL;
	int maxLength;
	int height, width;



	Matrix2() {};
	Matrix2(int height, int width, bool allocateHost);
	Matrix2(FillFunction& fillFunction, int height, int width);
	int e(int i, int j);
	float& operator()(int i, int j);
	void fill(FillFunction& fillFunction);
	void constantFill(float fh);
	void scale(float c);
	void sqrt(Matrix2& B, int num);
	void mean(Matrix2& mean);
	void variance(Matrix2& mean, Matrix2& variance);
	void normalize(Matrix2& mean, Matrix2& std, Matrix2& normalizedOutput);
	void print();
	void allocateHost();
	void deallocateHost();
	void copyToDevice();
	void copyToHost();
	void copy(float* matrix);
	void setDims(int height, int width);
	void setHeight(int height);
	void setWidth(int width);
	MatrixBatch subMatrixBatch(int numMatrices, int subHeight);

	static void add(Matrix2& A, Matrix2& B, Matrix2& C);
	static void elementMultiply(Matrix2& A, Matrix2& B, Matrix2& C);

	static void multiplyABC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyAtBC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyAtBtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyABtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);

	template<typename Function, typename... Params>
	static void runElementKernel(int height, int width, int sharedMemory, Function function, Params... params) {
		int N = height * width;
		int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		if (sharedMemory == 0) {
			function << < numBlocks, THREADS_PER_BLOCK >> > (forward<Params>(params)...);
		}
		else {
			function << < numBlocks, THREADS_PER_BLOCK, sharedMemory >> > (forward<Params>(params)...);
		}
	}
	template<typename Function, typename... Params>
	static void runRowKernel(int height, int width, int sharedMemory, Function function, Params... params) {
		if (sharedMemory == 0) {
			function << < height, THREADS_PER_BLOCK >> > (forward<Params>(params)...);
		}
		else {
			function << < height, THREADS_PER_BLOCK, sharedMemory >> > (forward<Params>(params)...);
		}
	}
	template<typename Function, typename... Params>
	static void runColumnKernel(int height, int width, int sharedMemory, Function function, Params... params) {
		if (sharedMemory == 0) {
			function << < width, THREADS_PER_BLOCK >> > (forward<Params>(params)...);
		}
		else {
			function << < width, THREADS_PER_BLOCK, sharedMemory >> > (forward<Params>(params)...);
		}
	}
};