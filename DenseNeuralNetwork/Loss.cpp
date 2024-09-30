#include "Loss.h"

Loss* Loss::MEAN_SQUARED_ERROR = { new MeanSquaredError() };
Loss* Loss::BINARY_CROSS_ENTROPY = { new BinaryCrossEntropy() };
Loss* Loss::CATEGORICAL_CROSS_ENTROPY = { new CategoricalCrossEntropy() };
Loss* Loss::ACCURACY = { new Accuracy() };
Loss* Loss::BINARY_ACCURACY = { new BinaryAccuracy() };
Loss* Loss::ALL_LOSSES[Loss::NUM_LOSSES] = {MEAN_SQUARED_ERROR, BINARY_CROSS_ENTROPY, CATEGORICAL_CROSS_ENTROPY};

double MeanSquaredError::loss(Layer* layer, double** yTrue) {
	double sum = 0;
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			sum += (layer->neurons[i][j] - yTrue[i][j]) * (layer->neurons[i][j] - yTrue[i][j]);
		}
	}
	return sum / layer->size;
}

void MeanSquaredError::differentiate(Layer* layer, double** yTrue) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			layer->neuronGradient[i][j] = 2 * (layer->neurons[i][j] - yTrue[i][j]) / layer->size;
		}
	}
}

string MeanSquaredError::toString() {
	string toString = "MeanSquaredError";
	return toString;
}

double BinaryCrossEntropy::loss(Layer* layer, double** yTrue) {
	double sum = 0;
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			sum += yTrue[i][j] * log(layer->neurons[i][j]) + (1 - yTrue[i][j]) * log(1 - layer->neurons[i][j]);
		}
	}
	return -sum / layer->size;
}

void BinaryCrossEntropy::differentiate(Layer* layer, double** yTrue) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			layer->neuronGradient[i][j] = (-yTrue[i][j] / layer->neurons[i][j] + (1 - yTrue[i][j]) / (1 - layer->neurons[i][j])) / layer->size;
		}
	}
}

string BinaryCrossEntropy::toString() {
	string toString = "BinaryCrossEntropy";
	return toString;
}

double CategoricalCrossEntropy::loss(Layer* layer, double** yTrue) {
	double sum = 0;
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			if (yTrue[i][j] != 0) {
				sum += yTrue[i][j] * log(layer->neurons[i][j] + 0.0000001);
			}
		}
	}
	return -sum / layer->size;
}

void CategoricalCrossEntropy::differentiate(Layer* layer, double** yTrue) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			if (yTrue[i][j] != 0) {
				layer->neuronGradient[i][j] = -yTrue[i][j] / (layer->size * layer->neurons[i][j] + 0.0000001);
			}
			else {
				layer->neuronGradient[i][j] = 0;
			}
		}
	}
}

string CategoricalCrossEntropy::toString() {
	string toString = "CategoricalCrossEntropy";
	return toString;
}

double Accuracy::loss(Layer* layer, double** yTrue) {
	int sum = 0;
	for (int i = 0; i < layer->batchSize; i++) {
		int predMax = 0;
		int trueMax = 0;
		for (int j = 0; j < layer->size; j++) {
			if (layer->neurons[i][j] > layer->neurons[i][predMax]) {
				predMax = j;
			} if (yTrue[i][j] > yTrue[i][trueMax]) {
				trueMax = j;
			}
		}
		sum += predMax == trueMax ? 1 : 0;
	}
	return sum;
}

void Accuracy::differentiate(Layer* layer, double** yTrue) {
	throw invalid_argument("Accuracy is a metric and cannot be used as a loss function");
}

string Accuracy::toString() {
	return "Accuracy";
}

double BinaryAccuracy::loss(Layer* layer, double** yTrue) {
	double sum = 0;
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			if ((layer->neurons[i][j] >= 0.5 && yTrue[i][j] >= 0.5) || (layer->neurons[i][j] < 0.5 && yTrue[i][j] < 0.5)) {
				sum++;
			}
		}
	}
	return sum / layer->size;
}

void BinaryAccuracy::differentiate(Layer* layer, double** yTrue) {
	throw invalid_argument("BinaryAccuracy is a metric and cannot be used as a loss function");
}

string BinaryAccuracy::toString() {
	string toString = "BinaryAccuracy";
	return toString;
}