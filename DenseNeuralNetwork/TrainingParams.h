#pragma once
#include <iostream>
#include <stdarg.h>

using namespace std;

class Loss1D;

class Optimizer;

class TrainingParams {

public:

	static TrainingParams* DEFAULT;
	const static int LEARNING_RATE = 0, BATCH_SIZE = 1, NUM_EPOCHS = 2, VAL_SPLIT = 3, OPTIMIZER = 4, VAL_SIZE = 5, VAL_NUM_TOKENS = 6, X_VAL = 7, Y_VAL = 8;

	tuple<float, int, int, float, Optimizer*, int, int*, void*, void*> data;

	TrainingParams(float learningRate, int batchSize, int numEpochs, float valSplit, Optimizer* optimizer, int valSize, int* valNumTokens, void* XVal, void* yVal);
	TrainingParams(tuple<float, int, int, float, Optimizer*, int, int*, void*, void*> data);

	template<int I = 0, typename T>
	TrainingParams* with(T value) {
		TrainingParams* params = { new TrainingParams(data) };
		params->edit<I>(value);
		return params;
	}
	
	template<int I = 0, typename T>
	T get() {
		return (T)std::get<I>(data);
	}

	TrainingParams* withValData(int valSize, int* valNumTokens, void* X, void* y);

private:
	template<int I = 0, typename T>
	void edit(T value) {
		if (I >= 5 && I <= 8) {
			throw invalid_argument("Cannot edit VAL_SIZE, X_VAL, or Y_VAL directly.  Use withValData() instead");
		}
		std::get<I>(data) = value;
	}

};

