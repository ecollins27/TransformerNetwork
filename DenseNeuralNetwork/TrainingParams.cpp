#include "TrainingParams.h"
#include "Optimizer.h"


TrainingParams* TrainingParams::DEFAULT = { new TrainingParams(0.0001, 32, 10, 0.1, Optimizer::ADEMAMIX) };


TrainingParams::TrainingParams(double learningRate, int batchSize, int numEpochs, double valSplit, Optimizer* optimizer) {
	data = (void**)malloc(NUM_PARAMETERS * sizeof(void*));
	data[0] = (double*)malloc(sizeof(double));
	*((double*)data[0]) = learningRate;
	data[1] = (int*)malloc(sizeof(int));
	*((int*)data[1]) = batchSize;
	data[2] = (int*)malloc(sizeof(int));
	*((int*)data[2]) = numEpochs;
	data[3] = (double*)malloc(sizeof(double));
	*((double*)data[3]) = valSplit;
	data[4] = optimizer;
}

TrainingParams* TrainingParams::with(const int index, void* value) {
	data[index] = value;
	return this;
}

void* TrainingParams::get(const int index) {
	int newIndex = index + 1;
	return data[index];
}