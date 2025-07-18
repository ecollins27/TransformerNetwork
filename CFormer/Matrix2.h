#pragma once
#include "MatrixBatch.h"
#include "FillFunction.h"
#include "Utils.h"
#include <cuda_runtime.h>
#include <xmmintrin.h>

class Matrix2 {

public:

	static float ALPHA;
	static float BETA0, BETA1;

	static int NUM_DEVICES;
	static int* DEVICE_LENGTHS;
	static float*** DEVICES;  //NUM_CORES x NUM_DEVICES x DEVICE_LENGTHS


	float* host;
	int length, maxLength;
	int height, width;
	int threadNum;

	Matrix2() {};
	Matrix2(int height, int width, int threadNum);
	Matrix2(FillFunction& fillFunction, int height, int width, int threadNum);
	void free();
	int e(int i, int j);
	float& operator()(int i, int j);
	void fill(FillFunction& fillFunction);
	void constantFill(float fh);
	void scale(float c);
	void sqrt(Matrix2& B);
	void print();
	void copyToBatchDevice(int deviceNum, int batchSize);
	void copyToBatchDevice(int deviceNum, int batchSize, int threadNum);
	void copyToDevice(int deviceNum);
	void copyToDevice(int deviceNum, int threadNum);
	void copyToHost(int deviceNum);
	void copyToHost(int deviceNum, int copyLength);
	void copyToHost(int deviceNum, int copyLength, int threadNum);
	void copy(float* matrix);
	void copy(int height, int width, float** matrix);
	void copy(Matrix2& B);
	void copyTo(Matrix2& B);
	void transpose(Matrix2& B);
	void setDims(int height, int width);
	void setHeight(int height);
	void setWidth(int width);
	MatrixBatch subMatrixBatch(int numMatrices, int subHeight);

	static void add(Matrix2& A, Matrix2& B, Matrix2& C);
	static void add(int width, Matrix2& A, Matrix2& B, Matrix2& C);
	static void elementMultiply(Matrix2& A, Matrix2& B, Matrix2& C);
	static void linearCombo(float c1, Matrix2& A, float c2, Matrix2& B, Matrix2& C);

	static void multiplyABC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyAtBC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyAtBtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyABtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);

	static Matrix2* allocateMatrixArray(FillFunction& fillFunction, int batchSize, int height, int width);
	static Matrix2* allocateMatrixArray(int batchSize, int height, int width);

	static long long allocateDevices();
};