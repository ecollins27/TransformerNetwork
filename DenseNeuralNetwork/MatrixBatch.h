#pragma once
#include "Matrix2.h"

class MatrixBatch {

public:
	Matrix2* matrices;
	bool saveTranspose;
	int batchSize;

	MatrixBatch(bool saveTranspose);

	float& operator()(int i, int j, int k);
	void setBatchSize(Matrix2::FillFunction* fillFunction, int batchSize, int height, int width);
};

