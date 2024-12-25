#pragma once
#include <iostream>
#include <sstream>
#include <cstdarg>
#include <random>
#include <xmmintrin.h>
#include <thread>
using namespace std;

class Matrix2 {

public:
	class FillFunction;
	static FillFunction* ZERO_FILL;
	static FillFunction* UNIT_NORMAL_FILL;
	static FillFunction* UNIT_UNIFORM_FILL;

	float** matrix;
	float** matrixTrans;
	bool saveTranspose;
	bool transposeUpdated = true;

	Matrix2(){}
	Matrix2(FillFunction* fillFunction, int height, int width, bool saveTranspose);
	Matrix2(float** matrix, float** matrixTrans);

	float& operator()(int i, int j);
	void calculateTranspose(int height, int width);
	void calculateMatrix(int height, int width);
	void scale(int height, int width, float c);
	void copy(int height, int width, Matrix2& to);
	void fill(FillFunction* fillFunction, int height, int width);
	void print(int height, int width);
	Matrix2 subMatrix(int i, int j, int height, int width);

	static Matrix2* allocateMatrixArray(Matrix2::FillFunction* fillFunction, int x1, int x2, int x3, bool saveTranspose);
	static Matrix2** allocateMatrixArray2D(Matrix2::FillFunction* fillFunction, int x1, int x2, int x3, int x4, bool saveTranspose);
	static float dotProduct(int n, float* a, float* b);
	static void multiplyABC(int m, int n, int p, Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyAtBC(int m, int n, int p, Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyAtBtC(int m, int n, int p, Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyABtC(int m, int n, int p, Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);

	static void elementAdd(int m, int n, Matrix2& A, Matrix2& B, Matrix2& C, float c1, float c2, bool overwrite);
	static void elementMultiply(int m, int n, Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);

	virtual class FillFunction {
	public:
		virtual float operator()() = 0;
	};

	class ConstantFill : public FillFunction {
	public:
		float value;
		ConstantFill(float value);
		float operator()();
	};

	class NormalFill : public FillFunction {
	public:
		default_random_engine generator;
		normal_distribution<float>* distribution;

		NormalFill(float mean, float stdDeviation);
		float operator()();
	};

	class UniformFill : public FillFunction {
	public:
		default_random_engine generator;
		uniform_real_distribution<float>* distribution;

		UniformFill(float lowerBound, float upperBound);
		float operator()();
	};
};

