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

	TrainingParams(double learningRate, int batchSize, int numEpochs, double valSplit, Optimizer* optimizer);
	TrainingParams(void** data);

	TrainingParams* with(const int index, void* value);
	
	void* get(const int index);

};

