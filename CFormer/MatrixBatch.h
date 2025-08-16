#pragma once
#include "FillFunction.h"
#include <iostream>
#include "Utils.h"

using namespace std;

class Matrix;

class MatrixBatch {

public:

	float** host = NULL;
	int length, maxLength;
	int batchSize, height, width;
	bool isLayerOutput = false;

	MatrixBatch() {};
	MatrixBatch(int batchSize, int height, int width);
	MatrixBatch(FillFunction& fillFunction, int batchSize, int height, int width);
	void free();
	int e(int i, int j);
	float& operator()(int i, int j, int k);
	void fill(FillFunction& fillFunction);
	void print();
	void setDims(int height, int width);
	void setHeight(int height);
	void setWidth(int width);

	static MatrixBatch* allocateMatrixBatchArray(FillFunction& fill, int arrayLength, int batchSize, int height, int width);
	static MatrixBatch* allocateMatrixBatchArray(int arrayLength, int batchSize, int height, int width);

};

