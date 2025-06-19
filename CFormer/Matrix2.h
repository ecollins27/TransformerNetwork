#pragma once
#include "MatrixBatch.h"
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
	Matrix2(int height, int width, bool allocateHost);
	Matrix2(FillFunction& fillFunction, int height, int width);
	int e(int i, int j);
	float& operator()(int i, int j);
	void fill(FillFunction& fillFunction);
	void constantFill(float fh);
	void scale(float c);
	void sqrt(Matrix2& B, int num);
	void mean(Matrix2& mean);
	void variance(Matrix2& mean, Matrix2& variance);
	void normalize(Matrix2& mean, Matrix2& std, Matrix2& normalizedOutput);
	void print();
	void allocateHost();
	void deallocateHost();
	void copyToDevice();
	void copyToHost();
	void copy(float* matrix);
	void setDims(int height, int width);
	void setHeight(int height);
	void setWidth(int width);
	MatrixBatch subMatrixBatch(int numMatrices, int subHeight);

	static void add(Matrix2& A, Matrix2& B, Matrix2& C);
	static void elementMultiply(Matrix2& A, Matrix2& B, Matrix2& C);

	static void multiplyABC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyAtBC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyAtBtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);
	static void multiplyABtC(Matrix2& A, Matrix2& B, Matrix2& C, bool overwrite);

	template<typename Function, typename... Params>
	static void runElementKernel(int height, int width, int sharedMemory, Function function, Params... params);
	template<typename Function, typename... Params>
	void runRowKernel(int height, int width, int sharedMemory, Function function, Params... params);
	template<typename Function, typename... Params>
	void runColumnKernel(int height, int width, int sharedMemory, Function function, Params... params);

	class FillFunction {
	public:
		virtual float operator()(int i, int j) { 
			return 0;
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

__global__
void kernelMean(float* input, float* output, int M, int N);

__global__
void kernelVariance(float* input, float* mean, float* output, int M, int N);

__global__
void kernelNormalize(float* matrix, float* mean, float* std, float* output, int N, int width);