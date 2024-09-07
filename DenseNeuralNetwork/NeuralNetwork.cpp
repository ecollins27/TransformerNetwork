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
	double* averages = NULL;
	averages = new double[params->numMetrics + 1];
	int trainingNum = (int)(numData * (1 - params->valSplit));
	trainingNum += params->batchSize - trainingNum % params->batchSize;
	for (int epoch = 0; epoch < params->numEpochs; epoch++) {
		shuffle(numData, X, y);
		for (int i = 0; i < params->numMetrics + 1; i++) {
			averages[i] = 0;
		}
		for (int i = 0; i < trainingNum; i++) {
			if (i % params->batchSize == 0) {
				applyGradients(params);
				printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, params->numEpochs, i + 1, trainingNum, averages[params->numMetrics] / (i + 1));
				for (int j = 0; j < params->numMetrics; j++) {
					printf("%s:%f  ", params->metrics[j]->toString().c_str(), averages[j] / (i + 1));
				}
			}
			fit(X[i], y[i], averages, params);
		}
		applyGradients(params);
		printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, params->numEpochs, trainingNum, trainingNum, averages[params->numMetrics] / trainingNum);
		for (int j = 0; j < params->numMetrics; j++) {
			printf("%s:%f  ", params->metrics[j]->toString().c_str(), averages[j] / trainingNum);
		}
		for (int i = 0; i < params->numMetrics + 1; i++) {
			averages[i] = 0;
		}
		for (int i = trainingNum; i < numData; i++) {
			forwardPropagate(X[i]);
			for (int j = 0; j < params->numMetrics; j++) {
				averages[j] += params->metrics[j]->loss(outputLayer, y[i]);
			}
			averages[params->numMetrics] += lossFunction->loss(outputLayer, y[i]);
		}
		printf("ValLoss:%f  ", averages[params->numMetrics] / (numData - trainingNum));
		for (int i = 0; i < params->numMetrics; i++) {
			printf("Val%s:%f  ", params->metrics[i]->toString().c_str(), averages[i] / (numData - trainingNum));
		}
		printf("\n");
	}
}