#include "TrainingParams.h"
#include "Optimizer.h"


TrainingParams* TrainingParams::DEFAULT = { new TrainingParams(0.00001, 32, 10, 0.1f, Optimizer::ADEMAMIX, 0, NULL, NULL, NULL) };


TrainingParams::TrainingParams(float learningRate, int batchSize, int numEpochs, float valSplit, Optimizer* optimizer, int valSize, int* valNumTokens, void* XVal, void* yVal) {
	data = make_tuple(learningRate, batchSize, numEpochs, valSplit, optimizer, valSize, valNumTokens, XVal, yVal);
}

TrainingParams::TrainingParams(tuple<float, int, int, float, Optimizer*, int, int*, void*, void*> data) {
	this->data = make_tuple(std::get<0>(data), std::get<1>(data), std::get<2>(data), std::get<3>(data), std::get<4>(data), std::get<5>(data), std::get<6>(data), std::get<7>(data), std::get<8>(data));
}

TrainingParams* TrainingParams::withValData(int valSize, int* valNumTokens, void* XVal, void* yVal) {
	TrainingParams* params = { new TrainingParams(data) };
	std::get<5>(params->data) = valSize;
	std::get<6>(params->data) = valNumTokens;
	std::get<7>(params->data) = XVal;
	std::get<8>(params->data) = yVal;
	return params;
}