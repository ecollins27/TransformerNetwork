#pragma once
#include "Matrix2.h"
class Matrix3D {

public:
	float*** matrix;

	Matrix3D();
	Matrix3D(Matrix2::FillFunction* fillFunction, int depth, int height, int width);

	float& operator()(int i, int j, int k);
};

