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
	data[4] = (Optimizer**)malloc(sizeof(Optimizer*));
	*((Optimizer**)data[4]) = optimizer;
}

TrainingParams::TrainingParams(void** d) {
	data = (void**)malloc(NUM_PARAMETERS * sizeof(void*));
	data[0] = (double*)malloc(sizeof(double));
	*((double*)data[0]) = *((double*)d[0]);
	data[1] = (int*)malloc(sizeof(int));
	*((int*)data[1]) = *((int*)d[1]);
	data[2] = (int*)malloc(sizeof(int));
	*((int*)data[2]) = *((int*)d[2]);
	data[3] = (double*)malloc(sizeof(double));
	*((double*)data[3]) = *((double*)d[3]);
	data[4] = (Optimizer**)malloc(sizeof(Optimizer*));
	*((Optimizer**)data[4]) = *((Optimizer**)d[4]);
}