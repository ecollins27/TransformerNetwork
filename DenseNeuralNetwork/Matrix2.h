#pragma once
#include <iostream>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

using namespace std;

class Matrix2 {

public:
	float* matrix;
	int maxHeight, maxWidth;
	int height, width;
	int* heightRange;
	int* widthRange;
	int* elementRange;

	Matrix2(int height, int width);
	int e(int i, int j);
	float& operator()(int i, int j);
	void print();
	void setDims(int height, int width);

	struct dotProduct {
		Matrix2& A;
		Matrix2& B;
		Matrix2& C;

		__host__ __device__
		void operator()(int i);
	};

	struct rowMultiply {
		Matrix2& A;
		Matrix2& B;
		Matrix2& C;

		__host__ __device__
		void operator()(int i);
	};

	static void multiplyABC(Matrix2& A, Matrix2& B, Matrix2& C);
};

