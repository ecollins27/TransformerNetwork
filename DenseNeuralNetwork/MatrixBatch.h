#pragma once
#include <iostream>
#include <cublas_v2.h>

using namespace std;

class MatrixBatch {

public:
	float** device;
	float** host = NULL;
	int maxHeight, maxWidth;
	int batchSize, height, width;

	static float ALPHA;
	static float BETA;
	static cublasHandle_t HANDLE;

	MatrixBatch(int batchSize, int height, int width);
	int e(int i, int j);
	float& operator()(int b, int i, int j);
	void print();
	void allocateHost();
	void deallocateHost();
	void copyToDevice();
	void copy(float** matrix);
	void setDims(int height, int width);

	static void multiplyABC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C);
};

