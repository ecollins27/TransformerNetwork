#include "Loss1D.h"

float MeanSquaredError1D::loss(Layer1D* layer, float** yTrue) {
	float sum = 0;
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			sum += (layer->neurons(i, j) - yTrue[i][j]) * (layer->neurons(i, j) - yTrue[i][j]);
		}
	}
	return sum / layer->size;
}

void MeanSquaredError1D::differentiate(Layer1D* layer, float** yTrue) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			layer->neuronGradient.r(i, j) = 2 * (layer->neurons(i, j) - yTrue[i][j]) / layer->size;
		}
	}
}

string MeanSquaredError1D::toString() {
	string toString = "MeanSquaredError";
	return toString;
}

float BinaryCrossEntropy1D::loss(Layer1D* layer, float** yTrue) {
	float sum = 0;
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			if (yTrue[i][j] != 0) {
				sum += yTrue[i][j] * log(layer->neurons(i, j) + 0.0000001);
			} if (1 - yTrue[i][j] != 0) {
				sum += (1 - yTrue[i][j]) * log(1 - layer->neurons(i, j) + 0.0000001);
			}
		}
	}
	return -sum / layer->size;
}

void BinaryCrossEntropy1D::differentiate(Layer1D* layer, float** yTrue) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			layer->neuronGradient.r(i, j) = (-yTrue[i][j] / (layer->neurons(i, j) + 0.0000001) + (1 - yTrue[i][j]) / (1 - layer->neurons(i, j) + 0.0000001)) / layer->size;
		}
	}
}

string BinaryCrossEntropy1D::toString() {
	string toString = "BinaryCrossEntropy";
	return toString;
}

float CategoricalCrossEntropy1D::loss(Layer1D* layer, float** yTrue) {
	float sum = 0;
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			if (yTrue[i][j] != 0) {
				sum += yTrue[i][j] * log(layer->neurons(i, j) + 0.0000001);
			}
		}
	}
	return -sum / layer->size;
}

void CategoricalCrossEntropy1D::differentiate(Layer1D* layer, float** yTrue) {
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			if (yTrue[i][j] != 0) {
				layer->neuronGradient.r(i, j) = -yTrue[i][j] / (layer->size * layer->neurons(i, j) + 0.0000001);
			}
			else {
				layer->neuronGradient.r(i, j) = 0;
			}
		}
	}
}

string CategoricalCrossEntropy1D::toString() {
	string toString = "CategoricalCrossEntropy";
	return toString;
}

float Accuracy1D::loss(Layer1D* layer, float** yTrue) {
	int sum = 0;
	for (int i = 0; i < layer->batchSize; i++) {
		int predMax = 0;
		int trueMax = 0;
		for (int j = 0; j < layer->size; j++) {
			if (layer->neurons(i, j) > layer->neurons(i, predMax)) {
				predMax = j;
			} if (yTrue[i][j] > yTrue[i][trueMax]) {
				trueMax = j;
			}
		}
		sum += predMax == trueMax ? 1 : 0;
	}
	return sum;
}

void Accuracy1D::differentiate(Layer1D* layer, float** yTrue) {
	throw invalid_argument("Accuracy is a metric and cannot be used as a loss function");
}

string Accuracy1D::toString() {
	return "Accuracy";
}

float BinaryAccuracy1D::loss(Layer1D* layer, float** yTrue) {
	float sum = 0;
	for (int i = 0; i < layer->batchSize; i++) {
		for (int j = 0; j < layer->size; j++) {
			if ((layer->neurons(i, j) >= 0.5 && yTrue[i][j] >= 0.5) || (layer->neurons(i, j) < 0.5 && yTrue[i][j] < 0.5)) {
				sum++;
			}
		}
	}
	return sum / layer->size;
}

void BinaryAccuracy1D::differentiate(Layer1D* layer, float** yTrue) {
	throw invalid_argument("BinaryAccuracy is a metric and cannot be used as a loss function");
}

string BinaryAccuracy1D::toString() {
	string toString = "BinaryAccuracy";
	return toString;
}