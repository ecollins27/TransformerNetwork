#include "Loss.h"

Loss* Loss::MEAN_SQUARED_ERROR = { new MeanSquaredError1D() };
Loss* Loss::BINARY_CROSS_ENTROPY = { new BinaryCrossEntropy1D() };
Loss* Loss::CATEGORICAL_CROSS_ENTROPY = { new CategoricalCrossEntropy1D() };
Loss* Loss::ACCURACY = { new Accuracy1D() };
Loss* Loss::BINARY_ACCURACY = { new BinaryAccuracy1D() };
Loss* Loss::ALL_LOSSES[Loss::NUM_LOSSES] = {MEAN_SQUARED_ERROR, BINARY_CROSS_ENTROPY, CATEGORICAL_CROSS_ENTROPY};

float MeanSquaredError1D::loss(Layer* layer, float** yTrue, int thread, bool layer1D) {
	float sum = 0;
	int height;
	Matrix neurons;
	if (layer1D) {
		neurons = ((Layer1D*)layer)->neurons;
		height = layer->batchSize;
	}
	else {
		neurons = ((Layer2D*)layer)->neurons[thread];
		height = ((Layer2D*)layer)->numTokens[thread];
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < layer->size; j++) {
			sum += (neurons(i, j) - yTrue[i][j]) * (neurons(i, j) - yTrue[i][j]);
		}
	}
	return sum / layer->size;
}

void MeanSquaredError1D::differentiate(Layer* layer, float** yTrue, int thread, bool layer1D) {
	Matrix neurons, neuronGradient;
	int height;
	if (layer1D) {
		neurons = ((Layer1D*)layer)->neurons;
		neuronGradient = ((Layer1D*)layer)->neuronGradient;
		height = layer->batchSize;
	}
	else {
		neurons = ((Layer2D*)layer)->neurons[thread];
		neuronGradient = ((Layer2D*)layer)->neuronGradient[thread];
		height = ((Layer2D*)layer)->numTokens[thread];
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < layer->size; j++) {
			neuronGradient(i, j) = 2 * (neurons(i, j) - yTrue[i][j]) / layer->size;
		}
	}
}

string MeanSquaredError1D::toString() {
	string toString = "MeanSquaredError";
	return toString;
}

float BinaryCrossEntropy1D::loss(Layer* layer, float** yTrue, int thread, bool layer1D) {
	float sum = 0;
	int height;
	Matrix neurons;
	if (layer1D) {
		neurons = ((Layer1D*)layer)->neurons;
		height = layer->batchSize;
	}
	else {
		neurons = ((Layer2D*)layer)->neurons[thread];
		height = ((Layer2D*)layer)->numTokens[thread];
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < layer->size; j++) {
			if (yTrue[i][j] != 0) {
				sum += yTrue[i][j] * log(neurons(i, j) + 0.0000001);
			} if (1 - yTrue[i][j] != 0) {
				sum += (1 - yTrue[i][j]) * log(1 - neurons(i, j) + 0.0000001);
			}
		}
	}
	return -sum / layer->size;
}

void BinaryCrossEntropy1D::differentiate(Layer* layer, float** yTrue, int thread, bool layer1D) {
	Matrix neurons, neuronGradient;
	int height;
	if (layer1D) {
		neurons = ((Layer1D*)layer)->neurons;
		neuronGradient = ((Layer1D*)layer)->neuronGradient;
		height = layer->batchSize;
	}
	else {
		neurons = ((Layer2D*)layer)->neurons[thread];
		neuronGradient = ((Layer2D*)layer)->neuronGradient[thread];
		height = ((Layer2D*)layer)->numTokens[thread];
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < layer->size; j++) {
			neuronGradient(i, j) = (-yTrue[i][j] / (neurons(i, j) + 0.0000001) + (1 - yTrue[i][j]) / (1 - neurons(i, j) + 0.0000001)) / layer->size;
		}
	}
}

string BinaryCrossEntropy1D::toString() {
	string toString = "BinaryCrossEntropy";
	return toString;
}

float CategoricalCrossEntropy1D::loss(Layer* layer, float** yTrue, int thread, bool layer1D) {
	float sum = 0;
	int height;
	Matrix neurons;
	if (layer1D) {
		neurons = ((Layer1D*)layer)->neurons;
		height = layer->batchSize;
	}
	else {
		neurons = ((Layer2D*)layer)->neurons[thread];
		height = ((Layer2D*)layer)->numTokens[thread];
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < layer->size; j++) {
			if (yTrue[i][j] != 0) {
				sum += yTrue[i][j] * log(neurons(i, j) + 0.0000001);
			}
		}
	}
	return -sum / layer->size;
}

void CategoricalCrossEntropy1D::differentiate(Layer* layer, float** yTrue, int thread, bool layer1D) {
	Matrix neurons, neuronGradient;
	int height;
	if (layer1D) {
		neurons = ((Layer1D*)layer)->neurons;
		neuronGradient = ((Layer1D*)layer)->neuronGradient;
		height = layer->batchSize;
	}
	else {
		neurons = ((Layer2D*)layer)->neurons[thread];
		neuronGradient = ((Layer2D*)layer)->neuronGradient[thread];
		height = ((Layer2D*)layer)->numTokens[thread];
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < layer->size; j++) {
			if (yTrue[i][j] != 0) {
				neuronGradient(i, j) = -yTrue[i][j] / (layer->size * neurons(i, j) + 0.0000001);
			}
			else {
				neuronGradient(i, j) = 0;
			}
		}
	}
}

string CategoricalCrossEntropy1D::toString() {
	string toString = "CategoricalCrossEntropy";
	return toString;
}

float Accuracy1D::loss(Layer* layer, float** yTrue, int thread, bool layer1D) {
	int sum = 0;
	int height;
	Matrix neurons;
	if (layer1D) {
		neurons = ((Layer1D*)layer)->neurons;
		height = layer->batchSize;
	}
	else {
		neurons = ((Layer2D*)layer)->neurons[thread];
		height = ((Layer2D*)layer)->numTokens[thread];
	}
	for (int i = 0; i < height; i++) {
		int predMax = 0;
		int trueMax = 0;
		for (int j = 0; j < layer->size; j++) {
			if (neurons(i, j) > neurons(i, predMax)) {
				predMax = j;
			} if (yTrue[i][j] > yTrue[i][trueMax]) {
				trueMax = j;
			}
		}
		sum += predMax == trueMax ? 1 : 0;
	}
	return sum;
}

void Accuracy1D::differentiate(Layer* layer, float** yTrue, int thread, bool layer1D) {
	throw invalid_argument("Accuracy is a metric and cannot be used as a loss function");
}

string Accuracy1D::toString() {
	return "Accuracy";
}

float BinaryAccuracy1D::loss(Layer* layer, float** yTrue, int thread, bool layer1D) {
	float sum = 0;
	int height;
	Matrix neurons;
	if (layer1D) {
		neurons = ((Layer1D*)layer)->neurons;
		height = layer->batchSize;
	}
	else {
		neurons = ((Layer2D*)layer)->neurons[thread];
		height = ((Layer2D*)layer)->numTokens[thread];
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < layer->size; j++) {
			if ((neurons(i, j) >= 0.5 && yTrue[i][j] >= 0.5) || (neurons(i, j) < 0.5 && yTrue[i][j] < 0.5)) {
				sum++;
			}
		}
	}
	return sum / layer->size;
}

void BinaryAccuracy1D::differentiate(Layer* layer, float** yTrue, int thread, bool layer1D) {
	throw invalid_argument("BinaryAccuracy is a metric and cannot be used as a loss function");
}

string BinaryAccuracy1D::toString() {
	string toString = "BinaryAccuracy";
	return toString;
}