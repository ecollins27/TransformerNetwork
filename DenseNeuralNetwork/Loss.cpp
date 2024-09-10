#include "Loss.h"

Loss* Loss::MEAN_SQUARED_ERROR = { new MeanSquaredError() };
Loss* Loss::BINARY_CROSS_ENTROPY = { new BinaryCrossEntropy() };
Loss* Loss::CATEGORICAL_CROSS_ENTROPY = { new CategoricalCrossEntropy() };
Loss* Loss::ACCURACY = { new Accuracy() };
Loss* Loss::BINARY_ACCURACY = { new BinaryAccuracy() };
Loss* Loss::ALL_LOSSES[3] = {MEAN_SQUARED_ERROR, BINARY_CROSS_ENTROPY, CATEGORICAL_CROSS_ENTROPY};

double MeanSquaredError::loss(Layer* layer, double* yTrue) {
	double sum = 0;
	for (int i = 0; i < layer->size; i++) {
		sum += (layer->neurons[i][0] - yTrue[i]) * (layer->neurons[i][0] - yTrue[i]);
	}
	return sum / layer->size;
}

void MeanSquaredError::differentiate(Layer* layer, double* yTrue) {
	for (int i = 0; i < layer->size; i++) {
		layer->neuronGradient[i][0] = 2 * (layer->neurons[i][0] - yTrue[i]) / layer->size;
	}
}

string MeanSquaredError::toString() {
	string toString = "MeanSquaredError";
	return toString;
}

double BinaryCrossEntropy::loss(Layer* layer, double* yTrue) {
	double sum = 0;
	for (int i = 0; i < layer->size; i++) {
		sum += yTrue[i] * log(layer->neurons[i][0]) + (1 - yTrue[i]) * log(1 - layer->neurons[i][0]);
	}
	return -sum / layer->size;
}

void BinaryCrossEntropy::differentiate(Layer* layer, double* yTrue) {
	for (int i = 0; i < layer->size; i++) {
		layer->neuronGradient[i][0] = (-yTrue[i] / layer->neurons[i][0] + (1 - yTrue[i]) / (1 - layer->neurons[i][0])) / layer->size;
	}
}

string BinaryCrossEntropy::toString() {
	string toString = "BinaryCrossEntropy";
	return toString;
}

double CategoricalCrossEntropy::loss(Layer* layer, double* yTrue) {
	double sum = 0;
	for (int i = 0; i < layer->size; i++) {
		sum += yTrue[i] * log(layer->neurons[i][0]);
	}
	return -sum / layer->size;
}

void CategoricalCrossEntropy::differentiate(Layer* layer, double* yTrue) {
	for (int i = 0; i < layer->size; i++) {
		layer->neuronGradient[i][0] = -yTrue[i] / (layer->size * layer->neurons[i][0]);
	}
}

string CategoricalCrossEntropy::toString() {
	string toString = "CategoricalCrossEntropy";
	return toString;
}

double Accuracy::loss(Layer* layer, double* yTrue) {
	int predMax = 0;
	int trueMax = 0;
	for (int i = 0; i < layer->size; i++) {
		if (layer->neurons[i][0] > layer->neurons[predMax][0]) {
			predMax = i;
		} if (yTrue[i] > yTrue[trueMax]) {
			trueMax = i;
		}
	}
	return predMax == trueMax ? 1 : 0;
}

void Accuracy::differentiate(Layer* layer, double* yTrue) {
	throw invalid_argument("Accuracy is a metric and cannot be used as a loss function");
}

string Accuracy::toString() {
	return "Accuracy";
}

double BinaryAccuracy::loss(Layer* layer, double* yTrue) {
	double sum = 0;
	for (int i = 0; i < layer->size; i++) {
		if ((layer->neurons[i][0] >= 0.5 && yTrue[i] >= 0.5) || (layer->neurons[i][0] < 0.5 && yTrue[i] < 0.5)) {
			sum++;
		}
	}
	return sum / layer->size;
}

void BinaryAccuracy::differentiate(Layer* layer, double* yTrue) {
	throw invalid_argument("BinaryAccuracy is a metric and cannot be used as a loss function");
}

string BinaryAccuracy::toString() {
	string toString = "BinaryAccuracy";
	return toString;
}