#pragma once
#include "Matrix.h"
class Matrix3D {

public:
	float*** matrix;

	Matrix3D(){}
	Matrix3D(Matrix::FillFunction* fillFunction, int depth, int height, int width);

	float& operator()(int i, int j, int k);
	bool containsIllegalValue(int depth, int height, int width);

	static Matrix3D* allocateMatrix3DArray(Matrix::FillFunction* fillFunction, int x1, int x2, int x3, int x4);
	static Matrix3D** allocateMatrix3DArray2D(Matrix::FillFunction* fillFunction, int x1, int x2, int x3, int x4, int x5);
	static void matrixTensorMultiply(int m, int n, int p, Matrix A, Matrix3D B, Matrix C, bool overwrite);
};

