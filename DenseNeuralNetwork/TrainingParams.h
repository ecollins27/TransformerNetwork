#pragma once
#include <iostream>
#include <stdarg.h>
#include "Dataset.h"

using namespace std;

class Loss1D;

class Optimizer;

class TrainingParams {

public:

	static TrainingParams* DEFAULT;
	const static int LEARNING_RATE = 0, BATCH_SIZE = 1, NUM_EPOCHS = 2, VAL_SPLIT = 3, OPTIMIZER = 4, VAL_DATA = 5;;

	tuple<float, int, int, float, Optimizer*, Dataset*> data;

	TrainingParams(float learningRate, int batchSize, int numEpochs, float valSplit, Optimizer* optimizer, Dataset* valData);
	TrainingParams(tuple<float, int, int, float, Optimizer*, Dataset*> data);

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

private:
	template<int I = 0, typename T>
	void edit(T value) {
		std::get<I>(data) = value;
	}

};

