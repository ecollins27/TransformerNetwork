#pragma once
#include "FillFunction.h"
#include <iostream>
#include <cublas_v2.h>

using namespace std;

class Matrix2;

class MatrixBatch {

public:
	static float ALPHA;
	static float BETA0, BETA1;
	static const int THREADS_PER_BLOCK = 256;
	static cublasHandle_t HANDLE;


	float** device;
	float** hostDevice;
	float** host = NULL;
	int length, maxLength;
	int batchSize, height, width;



	MatrixBatch() {};
	MatrixBatch(int batchSize, int height, int width, bool allocateHost);
	MatrixBatch(FillFunction& fillFunction, int batchSize, int height, int width);
	int e(int i, int j);
	float& operator()(int i, int j, int k);
	void fill(FillFunction& fillFunction);
	void constantFill(float fh);
	void scale(float c);
	void sqrt(MatrixBatch& B);
	void condense(Matrix2& B);
	void print();
	void allocateHost();
	void deallocateHost();
	void copyToDevice();
	void copyToHost();
	void copy(float* matrix);
	void copy(float** matrix);
	void copy(MatrixBatch& B);
	void setDims(int height, int width);
	void setHeight(int height);
	void setWidth(int width);

	static void add(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C);
	static void elementMultiply(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C);
	static void linearCombo(float c1, MatrixBatch& A, float c2, MatrixBatch& B, MatrixBatch& C);

	static void multiplyABC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite);
	static void multiplyABC(Matrix2& A, MatrixBatch& B, MatrixBatch& C, bool overwrite);
	static void multiplyAtBC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite);
	static void multiplyAtBC(Matrix2& A, MatrixBatch& B, MatrixBatch& C, bool overwrite);
	static void multiplyAtBtC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite);
	static void multiplyABtC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite);

	static MatrixBatch* allocateMatrixBatchArray(FillFunction& fill, int arrayLength, int batchSize, int height, int width);
	static MatrixBatch* allocateMatrixBatchArray(int arrayLength, int batchSize, int height, int width, bool allocateHost);
};

