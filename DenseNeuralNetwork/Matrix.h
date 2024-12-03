#pragma once
#include <iostream>
#include <sstream>
#include <cstdarg>
#include <random>
#include <xmmintrin.h>
#include <thread>
using namespace std;



const class Matrix {

public:

	class FillFunction;
	static FillFunction* ZERO_FILL;
	static FillFunction* UNIT_NORMAL_FILL;
	static FillFunction* UNIT_UNIFORM_FILL;

	static float dotProduct(int n, float* x, float* y);
	static float** allocateMatrix(FillFunction* fillFunction, int height, int width);
	static float*** allocate3DMatrix(FillFunction* fillFunction, int d1, int d2, int d3);
	static float**** allocate4DMatrix(FillFunction* fillFunction, int d1, int d2, int d3, int d4);
	static void deallocateMatrix(float** A, int height, int width);
	static void deallocate3DMatrix(float*** A, int d1, int d2, int d3);
	static void deallocate4DMatrix(float**** A, int d1, int d2, int d3, int d4);
	static bool containsNaN(int height, int width, float** A);
	static void add(int m, int n, float** A, float** B, float** C, float scalar1, float scalar2);
	static void scale(int m, int n, float** A, float scalar);
	static void transpose(int m, int n, float** A, float** At);
	static void transposeInPlace(int m, float** A);
	static void matrixMultiplyABC(int m, int n, int p, float** A, float** B, float** C, bool overwrite);
	static void matrixMultiplyAtBC(int m, int n, int p, float** A, float** B, float** C, bool overwrite);
	//static void rowMultiplyABtC(int n, int p, float** A, float** B, float** C, bool overwrite);
	static void subMatrixMultiplyABtC(int m, int n, int p, float** A, float** B, float** C, bool overwrite, int startY);
	static void matrixMultiplyABtC(int m, int n, int p, float** A, float** B, float** C, bool overwrite);
	//static void rowMultiplyABtCt(int n, int p, float** A, float** B, float** C, bool overwrite);
	static void matrixMultiplyABtCt(int m, int n, int p, float** A, float** B, float** C, bool overwrite);
	static void matrixTensorMultiply(int m, int n, int p, float** A, float*** B, float** C, bool overwrite);
	static void elementMultiply(int m, int n, float** A, float** B, float** C, bool overwrite);
	static void fill(FillFunction* fillFunction, int m, int n, float** A);
	static void copy(int m, int n, float** from, float** to);
	static void print(int m, int n, float** A);

	static virtual class FillFunction {
	public:
		virtual float get() = 0;
	};

	static class ConstantFill : public FillFunction {
	public:
		float value;
		ConstantFill(float value);
		float get();
	};

	static class NormalFill : public FillFunction {
	public:
		default_random_engine generator;
		normal_distribution<float>* distribution;

		NormalFill(float mean, float stdDeviation);
		float get();
	};

	static class UniformFill : public FillFunction {
	public:
		default_random_engine generator;
		uniform_real_distribution<float>* distribution;

		UniformFill(float lowerBound, float upperBound);
		float get();
	};
};

