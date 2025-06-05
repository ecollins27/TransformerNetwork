#pragma once
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <xmmintrin.h>
#include <random>

using namespace std;

class Matrix2 {

public:
	class FillFunction;

	static float ALPHA;
	static float BETA0, BETA1;
	static int THREADS_PER_BLOCK;
	static cublasHandle_t HANDLE;


	float* device;
	float* host = NULL;
	int maxLength;
	int height, width;


	Matrix2() {};
	Matrix2(int height, int width);
	int e(int i, int j);
	float& operator()(int i, int j);
	void fill(FillFunction& fillFunction);
	void constantFill(float fh);
	void scale(float c);
	void sqrt(Matrix2& B, int num);
	void print();
	void allocateHost();
	void deallocateHost();
	void copyToDevice();
	void copyToHost();
	void copy(float* matrix);
	void setDims(int height, int width);
	void setHeight(int height);
	void setWidth(int width);

	static void add(Matrix2& A, Matrix2& B, Matrix2& C);
	static void simdAdd(Matrix2& A, Matrix2& B, Matrix2& C);
	static void elementMultiply(Matrix2& A, Matrix2& B, Matrix2& C);

	static void multiplyABC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyAtBC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyAtBtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyABtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);

	class FillFunction {
	public:
		virtual float operator()(int i, int j) { 
			return 0.1;
		};
	};

	class ConstantFill : public FillFunction {
	public:
		float value;
		ConstantFill(float value);
		float operator()(int i, int j) override;
	};

	class NormalFill : public FillFunction {
	public:
		default_random_engine generator;
		normal_distribution<float>* distribution;

		NormalFill(float mean, float stdDeviation);
		float operator()(int i, int j) override;
	};

	class UniformFill : public FillFunction {
	public:
		default_random_engine generator;
		uniform_real_distribution<float>* distribution;

		UniformFill(float lowerBound, float upperBound);
		float operator()(int i, int j) override;
	};

	static ConstantFill ZERO_FILL;
	static NormalFill UNIT_NORMAL_FILL;
	static UniformFill UNIT_UNIFORM_FILL;
};

__global__
void kernelAdd(int N, float* A, float* B, float* C);

__global__
void kernelMultiply(int N, float* A, float* B, float* C);