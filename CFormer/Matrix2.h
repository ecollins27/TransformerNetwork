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
	static const int THREADS_PER_BLOCK = 256;
	static cublasHandle_t HANDLE;


	float* device;
	float* host = NULL;
	float** batchDevice = NULL;
	int length, maxLength;
	int height, width;



	Matrix2() {};
	Matrix2(int height, int width, bool allocateHost);
	Matrix2(FillFunction& fillFunction, int height, int width);
	~Matrix2();
	int e(int i, int j);
	float& operator()(int i, int j);
	void fill(FillFunction& fillFunction);
	void constantFill(float fh);
	void scale(float c);
	void sqrt(Matrix2& B);
	void print();
	void allocateHost();
	void deallocateHost();
	void allocateBatchDevice(int batchSize);
	void deallocateBatchDevice(int batchSize);
	void copyToDevice();
	void copyToHost();
	void copy(float* matrix);
	void copy(float** matrix);
	void copy(Matrix2& B);
	void transpose(Matrix2& B, int N);
	void setDims(int height, int width);
	void setHeight(int height);
	void setWidth(int width);
	MatrixBatch subMatrixBatch(int numMatrices, int subHeight);

	static void add(Matrix2& A, Matrix2& B, Matrix2& C);
	static void add(int width, Matrix2& A, Matrix2& B, Matrix2& C);
	static void elementMultiply(Matrix2& A, Matrix2& B, Matrix2& C);
	static void linearCombo(float c1, Matrix2& A, float c2, Matrix2& B, Matrix2& C);

	static void multiplyABC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyAtBC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyAtBtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyABtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);

	static Matrix2* allocateMatrixArray(FillFunction& fillFunction, int batchSize, int height, int width);
	static Matrix2* allocateMatrixArray(int batchSize, int height, int width, bool allocateHost);

};