#include "TransformerModel.h"

TransformerModel::TransformerModel(int inputSize) {
	inputLayer = { new InputLayer(inputSize) };
	outputLayer = inputLayer;
	t = 0;
}

void TransformerModel::predict(float** input) {
	inputLayer->setInput(input);
	inputLayer->predict();
}

void TransformerModel::forwardPropagate(float** input) {
	inputLayer->setInput(input);
	inputLayer->forwardPropagate();
}

void TransformerModel::backPropagate(Loss* lossFunction, float* yTrue) {
	lossFunction->differentiate(outputLayer, &yTrue);
	outputLayer->backPropagate();
}

void TransformerModel::fit(Loss* lossFunction, float** X, float* y, float* losses, int numMetrics, Loss** metrics, TrainingParams* params) {
	forwardPropagate(X);
	backPropagate(lossFunction, y);
	for (int i = 0; i < numMetrics; i++) {
		losses[i] += metrics[i]->loss(outputLayer, &y);
	}
	losses[numMetrics] += lossFunction->loss(outputLayer, &y);
}

void TransformerModel::fit(Loss* lossFunction, int numData, float*** X, float** y, int numMetrics, Loss** metrics, TrainingParams* params) {

}

void TransformerModel::test(Loss* lossFunction, int numData, float*** X, float** y, int numMetrics, Loss** metrics) {

}

void TransformerModel::shuffle(int numData, float*** X, float** y) {

}

void TransformerModel::save(string fileName) {

}