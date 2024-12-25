#pragma once
#include "Matrix3D.h"

class Matrix3DBatch {

public:
	Matrix3D* matrices;

	Matrix3DBatch(){}
	Matrix3DBatch(Matrix2::FillFunction* fillFunction, int batchSize, int depth, int height, int width);

	float& operator()(int i, int j, int k, int l);

	Matrix3D& operator[](int i);
};

