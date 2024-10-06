#pragma once
#include <iostream>
#include <sstream>
#include <cstdarg>
#include <random>
using namespace std;



const class Matrix {

public:

	class FillFunction;

	static FillFunction* ZERO_FILL;
	static FillFunction* UNIT_NORMAL_FILL;
	static FillFunction* UNIT_UNIFORM_FILL;

	static double** allocateMatrix(FillFunction* fillFunction, int height, int width);
	static void deallocateMatrix(double** A, int height, int width);
	static void add(int m, int n, double** A, double** B, double** C, double scalar1, double scalar2);
	static void scale(int m, int n, double** A, double scalar);
	static void matrixMultiplyABC(int m, int n, int p, double** A, double** B, double** C, bool overwrite);
	static void matrixMultiplyAtBC(int m, int n, int p, double** A, double** B, double** C, bool overwrite);
	static void matrixMultiplyABtC(int m, int n, int p, double** A, double** B, double** C, bool overwrite);
	static void matrixMultiplyABCt(int m, int n, int p, double** A, double** B, double** C, bool overwrite);
	static void matrixMultiplyAtBtC(int m, int n, int p, double** A, double** B, double** C, bool overwrite);
	static void matrixTensorMultiply(int m, int n, int p, double** A, double*** B, double** C, bool overwrite);
	static void elementMultiply(int m, int n, double** A, double** B, double** C, bool overwrite);
	static void fill(FillFunction* fillFunction, int m, int n, double** A);
	static void copy(int m, int n, double** from, double** to);
	static void print(int m, int n, double** A);

	static virtual class FillFunction {
	public:
		virtual double get() = 0;
	};

	static class ConstantFill : public FillFunction {
	public:
		double value;
		ConstantFill(double value);
		double get();
	};

	static class NormalFill : public FillFunction {
	public:
		default_random_engine generator;
		normal_distribution<double>* distribution;

		NormalFill(double mean, double stdDeviation);
		double get();
	};

	static class UniformFill : public FillFunction {
	public:
		default_random_engine generator;
		uniform_real_distribution<double>* distribution;

		UniformFill(double lowerBound, double upperBound);
		double get();
	};
};

