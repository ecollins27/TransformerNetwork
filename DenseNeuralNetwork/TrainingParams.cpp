#include "TrainingParams.h"
#include "Optimizer.h"

TrainingParams* TrainingParams::DEFAULT = { new TrainingParams(0.0001, 32, 10, 0.1, Optimizer::ADEMAMIX, 0, NULL) };

TrainingParams::TrainingParams(double learningRate, int batchSize, int numEpochs, double valSplit, Optimizer* optimizer, int numMetrics, Loss** metrics) {
	this->learningRate = learningRate;
	this->batchSize = batchSize;
	this->numEpochs = numEpochs;
	this->valSplit = valSplit;
	this->numMetrics = numMetrics;
	this->optimizer = optimizer;
	this->metrics = metrics;
}

TrainingParams* TrainingParams::withLearningRate(double learningRate) {
	return { new TrainingParams(learningRate, batchSize, numEpochs, valSplit, optimizer, numMetrics, metrics) };
}

TrainingParams* TrainingParams::withBatchSize(int batchSize) {
	return { new TrainingParams(learningRate, batchSize, numEpochs, valSplit, optimizer, numMetrics, metrics) };
}

TrainingParams* TrainingParams::withNumEpochs(int numEpochs) {
	return { new TrainingParams(learningRate, batchSize, numEpochs, valSplit, optimizer, numMetrics, metrics) };
}

TrainingParams* TrainingParams::withOptimizer(Optimizer* optimizer) {
	return { new TrainingParams(learningRate, batchSize, numEpochs, valSplit, optimizer, numMetrics, metrics) };
}

TrainingParams* TrainingParams::withMetrics(int numMetrics, ... ) {
	va_list arguments;
	va_start(arguments, numMetrics);
	Loss** newMetrics = (Loss**)malloc(numMetrics * sizeof(Loss*));
	for (int i = 0; i < numMetrics; i++) {
		newMetrics[i] = va_arg(arguments, Loss*);
	}
	return { new TrainingParams(learningRate, batchSize, numEpochs, valSplit, optimizer, numMetrics, newMetrics) };
}

TrainingParams* TrainingParams::withValidationSplit(double valSplit) {
	return { new TrainingParams(learningRate, batchSize, numEpochs, valSplit, optimizer, numMetrics, metrics) };
}