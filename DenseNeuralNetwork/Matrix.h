#pragma once
#include <iostream>
#include <sstream>
#include <cstdarg>
#include <random>
#include <xmmintrin.h>
#include <thread>
using namespace std;

class Matrix {

public:
	class FillFunction;
	static FillFunction* ZERO_FILL;
	static FillFunction* UNIT_NORMAL_FILL;
	static FillFunction* UNIT_UNIFORM_FILL;
	int maxHeight, maxWidth;

	float** matrix;
	float** matrixTrans;
	bool saveTranspose;
	bool* transposeUpdated;

	Matrix(){}
	Matrix(FillFunction* fillFunction, int height, int width, bool saveTranspose);
	Matrix(int maxHeight, int maxWidth, float** matrix, float** matrixTrans);

	void free();
	float operator()(int i, int j);
	float& r(int i, int j);
	void calculateTranspose(int height, int width);
	void calculateMatrix(int height, int width);
	void scale(int height, int width, float c);
	void copy(int height, int width, Matrix& to);
	void fill(FillFunction* fillFunction, int height, int width);
	void constantFill(float f, int height, int width);
	void sqrt(int m, int n, Matrix& B, int num);
	bool equals(int height, int width, Matrix A);
	bool similiar(int height, int width, Matrix A, float errorThreshold);
	void print(int height, int width);
	Matrix subMatrix(int i, int j, int height, int width);
	bool containsIllegalValue(int height, int width);
	void free(int height, int width);

	static float** allocateMatrix(Matrix::FillFunction* fillFunction, int height, int width);
	static void deallocateMatrix(float** matrix, int height, int width);
	static float*** allocateMatrix3D(Matrix::FillFunction* fillFunction, int depth, int height, int width);
	static Matrix* allocateMatrixArray(Matrix::FillFunction* fillFunction, int x1, int x2, int x3, bool saveTranspose);
	static Matrix** allocateMatrixArray2D(Matrix::FillFunction* fillFunction, int x1, int x2, int x3, int x4, bool saveTranspose);
	static float dotProduct(int n, float* a, float* b);
	static void multiplyABC(int m, int n, int p, Matrix& A, Matrix& B, Matrix& C, bool overwrite);
	static void multiplyAtBC(int m, int n, int p, Matrix& A, Matrix& B, Matrix& C, bool overwrite);
	static void multiplyAtBtC(int m, int n, int p, Matrix& A, Matrix& B, Matrix& C, bool overwrite);
	static void multiplyABtC(int m, int n, int p, Matrix& A, Matrix& B, Matrix& C, bool overwrite);

	static void add(int m, int n, Matrix& A, Matrix& B, Matrix& C);
	static void linearCombo(int m, int n, float c1, Matrix& A, float c2, Matrix& B, Matrix& C);
	static void elementMultiply(int m, int n, Matrix& A, Matrix& B, Matrix& C);
	static void calculateMean(int m, int n, Matrix& A, Matrix& mean, int num);
	static void calculateVariance(int m, int n, Matrix& A, Matrix& mean, Matrix& variance, int num);
	static void normalize(int m, int n, Matrix& A, Matrix& B, Matrix& mean, Matrix& std, int num);
	static void parameterNormalize(int m, int n, Matrix& A, Matrix& B, Matrix& mean, Matrix& std, Matrix& parameters, int num);

	class FillFunction {
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

