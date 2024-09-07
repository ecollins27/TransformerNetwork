#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(Loss* lossFunction, int inputSize) {
	this->lossFunction = lossFunction;
	inputLayer = { new InputLayer(inputSize) };
	outputLayer = inputLayer;
	t = 0;
}

void NeuralNetwork::addLayer(Layer* layer) {
	outputLayer->setNextLayer(layer);
	layer->setPrevLayer(outputLayer);
	outputLayer = layer;
}

void NeuralNetwork::forwardPropagate(double* input) {
	inputLayer->setInput(input);
	inputLayer->forwardPropagate();
}

double NeuralNetwork::getLoss(double* output) {
	return lossFunction->loss(outputLayer, output);
}

void NeuralNetwork::backPropagate(double* yTrue) {
	lossFunction->differentiate(outputLayer, yTrue);
	outputLayer->backPropagate();
}

void NeuralNetwork::applyGradients(TrainingParams* params) {
	t++;
	inputLayer->applyGradients(params, t);
}

void NeuralNetwork::fit(double* X, double* y, double* losses, TrainingParams* params) {
	forwardPropagate(X);
	backPropagate(y);
	for (int i = 0; i < params->numMetrics; i++) {
		losses[i] += params->metrics[i]->loss(outputLayer, y);
	}
	losses[params->numMetrics] += lossFunction->loss(outputLayer, y);
}

void NeuralNetwork::shuffle(int numData, double** X, double** y) {
	for (int i = 0; i < numData; i++) {
		int index = (int)(numData * ((double)rand() / (RAND_MAX + 1)));
		swap(X[i], X[index]);
		swap(y[i], y[index]);
	}
}

void NeuralNetwork::fit(int numData, double** X, double** y, TrainingParams* params) {
	double* averages = new double[params->numMetrics + 1];
	for (int epoch = 0; epoch < params->numEpochs; epoch++) {
		shuffle(numData, X, y);
		for (int i = 0; i < params->numMetrics + 1; i++) {
			averages[i] = 0;
		}
		for (int i = 0; i < numData; i++) {
			if (i % params->batchSize == 0) {
				applyGradients(params);
				printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, params->numEpochs, i + 1, numData, averages[params->numMetrics] / (i + 1));
				for (int j = 0; j < params->numMetrics; j++) {
					printf("%s:%f  ", params->metrics[j]->toString().c_str(), averages[j] / (i + 1));
				}
			}
			fit(X[i], y[i], averages, params);
		}
		printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, params->numEpochs, numData, numData, averages[params->numMetrics] / numData);
		for (int j = 0; j < params->numMetrics; j++) {
			printf("%s:%f  ", params->metrics[j]->toString().c_str(), averages[j] / numData);
		}
		printf("\n");
	}
}