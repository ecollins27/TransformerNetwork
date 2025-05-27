#pragma once
#include <iostream>
#include <cublas_v2.h>

#define ON_DEVICE true

using namespace std;

class Matrix2 {

public:
	float* matrix;
	int maxHeight, maxWidth;
	int height, width;

	Matrix2(int height, int width);
	int e(int i, int j);
	float& operator()(int i, int j);
	void print();
	void setDims(int height, int width);

	static void multiplyABC(Matrix2& A, Matrix2& B, Matrix2& C);
};

