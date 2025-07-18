#pragma once
#include "FillFunction.h"
#include <iostream>
#include "Utils.h"

using namespace std;

class Matrix2;

class MatrixBatch {

public:
	static float ALPHA;
	static float BETA0, BETA1;

	static int NUM_DEVICES;
	static int* DEVICE_BATCHSIZES;
	static int* DEVICE_LENGTHS;
	static float**** DEVICES; // NUM_CORES x NUM_DEVICES x DEVICE_BATCHSIZES x DEVICE_LENGTHS
	static float**** HOST_DEVICES;  // NUM_CORES x NUM_DEVICES x DEVICE_BATCHSIZES x DEVICE_LENGTHS

	float** host = NULL;
	int length, maxLength;
	int batchSize, height, width;
	int threadNum;

	MatrixBatch() {};
	MatrixBatch(int batchSize, int height, int width, int threadNum);
	MatrixBatch(FillFunction& fillFunction, int batchSize, int height, int width, int threadNum);
	void free();
	int e(int i, int j);
	float& operator()(int i, int j, int k);
	void fill(FillFunction& fillFunction);
	void constantFill(float fh);
	void scale(float c);
	void sqrt(MatrixBatch& B);
	void condense(Matrix2& B);
	void print();
	void copyToDevice(int deviceNum);
	void copyToDevice(int deviceNum, int threadNum);
	void copyToHost(int deviceNum);
	void copyToHost(int deviceNum, int copyLength);
	void copyToHost(int deviceNum, int copyLength, int threadNum);
	void copy(float* matrix);
	void copy(float** matrix);
	void copy(MatrixBatch& B);
	void copyTo(MatrixBatch& B);
	void setDims(int height, int width);
	void setHeight(int height);
	void setWidth(int width);

	static void allocateDevices(int numDevices, int* batchSizes, int* devices);
	static void add(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C);
	static void elementMultiply(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C);
	static void linearCombo(float c1, MatrixBatch& A, float c2, MatrixBatch& B, MatrixBatch& C);

	static void multiplyABC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite);
	static void multiplyABC(Matrix2& A, MatrixBatch& B, MatrixBatch& C, bool overwrite);
	static void multiplyAtBC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite);
	static void multiplyAtBC(Matrix2& A, MatrixBatch& B, MatrixBatch& C, bool overwrite);
	static void multiplyAtBtC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite);
	static void multiplyABtC(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C, bool overwrite);

	static MatrixBatch* allocateMatrixBatchArray(FillFunction& fill, int arrayLength, int batchSize, int height, int width);
	static MatrixBatch* allocateMatrixBatchArray(int arrayLength, int batchSize, int height, int width);

	static long long allocateDevices();
};

