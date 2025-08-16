#pragma once
#include "MatrixBatch.h"
#include "FillFunction.h"
#include "Utils.h"
#include <cuda_runtime.h>
#include <xmmintrin.h>

class Matrix {

public:


	float* host;
	int length, maxLength;
	int height, width;
	bool isLayerOutput = false;

	Matrix() {};
	Matrix(int height, int width);
	Matrix(FillFunction& fillFunction, int height, int width);
	void free();
	int e(int i, int j);
	float& operator()(int i, int j);
	void fill(FillFunction& fillFunction);
	void print();
	void setDims(int height, int width);
	void setHeight(int height);
	void setWidth(int width);
	void setLayerOutput(bool isLayerOutput);
	MatrixBatch subMatrixBatch(int numMatrices, int subHeight);

	static Matrix* allocateMatrixArray(FillFunction& fillFunction, int batchSize, int height, int width);
	static Matrix* allocateMatrixArray(int batchSize, int height, int width);
};