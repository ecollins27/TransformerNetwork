#include "TrainingParams.h"
#include "Optimizer.h"


TrainingParams* TrainingParams::DEFAULT = { new TrainingParams(0.00001f, 32, 10, 0.1f, Optimizer::ADEMAMIX, NULL) };


TrainingParams::TrainingParams(float learningRate, int batchSize, int numEpochs, float valSplit, Optimizer* optimizer, Dataset* valData) {
	data = make_tuple(learningRate, batchSize, numEpochs, valSplit, optimizer, valData);
}

TrainingParams::TrainingParams(tuple<float, int, int, float, Optimizer*, Dataset*> data) {
	this->data = make_tuple(std::get<0>(data), std::get<1>(data), std::get<2>(data), std::get<3>(data), std::get<4>(data), std::get<5>(data));
}