#pragma once
#include <iostream>
#include <stdarg.h>

using namespace std;

class Loss;

class Optimizer;

class TrainingParams {

public:

	static TrainingParams* DEFAULT;

	double learningRate;
	int batchSize;
	int numEpochs;
	int numMetrics;
	Optimizer* optimizer;
	Loss** metrics;

	TrainingParams(double learningRate, int batchSize, int numEpochs, Optimizer* optimizer, int numMetrics, Loss** metrics);
	TrainingParams* withLearningRate(double learningRate);
	TrainingParams* withBatchSize(int batchSize);
	TrainingParams* withNumEpochs(int numEpochs);
	TrainingParams* withMetrics(int numMetrics, ...);
	TrainingParams* withOptimizer(Optimizer* optimizer);
};

