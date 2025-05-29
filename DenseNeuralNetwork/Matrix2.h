#pragma once
#include <iostream>
#include <cublas_v2.h>


using namespace std;

class Matrix2 {

public:
	float* device;
	float* host = NULL;
	int maxHeight, maxWidth;
	int height, width;

	static float ALPHA;
	static float BETA;
	static cublasHandle_t HANDLE;
	Matrix2(int height, int width);
	int e(int i, int j);
	float& operator()(int i, int j);
	void print();
	void toHost();
	void copy(float* matrix);
	void setDims(int height, int width);

	static void multiplyABC(Matrix2& A, Matrix2& B, Matrix2& C);
};