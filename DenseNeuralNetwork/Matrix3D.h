#pragma once
#include "Matrix2.h"
class Matrix3D {

public:
	float*** matrix;

	Matrix3D(){}
	Matrix3D(Matrix2::FillFunction* fillFunction, int depth, int height, int width);

	float& operator()(int i, int j, int k);

	static void matrixTensorMultiply(int m, int n, int p, Matrix2 A, Matrix3D B, Matrix2 C, bool overwrite);
};

