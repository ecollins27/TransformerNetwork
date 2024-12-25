#pragma once
#include "Matrix2.h"

class MatrixBatch {

public:
	Matrix2* matrices;
	bool saveTranspose;
	int batchSize;

	MatrixBatch(){}
	MatrixBatch(int batchSize);
	MatrixBatch(Matrix2::FillFunction* fillFunction, int batchSize, int height, int width, bool saveTranspose);

	float& operator()(int i, int j, int k);
	Matrix2& operator[](int i);
};

