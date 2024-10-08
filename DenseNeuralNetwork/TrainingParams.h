#pragma once
#include <iostream>
#include <stdarg.h>

using namespace std;

class Loss;

class Optimizer;

class TrainingParams {

public:

	static TrainingParams* DEFAULT;
	const static int NUM_PARAMETERS = 5;
	const static int LEARNING_RATE = 0, BATCH_SIZE = 1, NUM_EPOCHS = 2, VAL_SPLIT = 3, OPTIMIZER = 4;

	void** data;

	TrainingParams(float learningRate, int batchSize, int numEpochs, float valSplit, Optimizer* optimizer);
	TrainingParams(void** data);

	template<typename T>
	TrainingParams* with(const int index, T value) {
		TrainingParams* params = { new TrainingParams(data) };
		params->edit(index, value);
		return params;
	}
	
	template<typename T>
	T get(const int index) {
		return *((T*)data[index]);
	}

private:
	template<typename T>
	void edit(const int index, T value) {
		*((T*)data[index]) = value;
	}

};

