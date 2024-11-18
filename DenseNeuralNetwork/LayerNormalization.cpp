#include "LayerNormalization.h"

LayerNormalization::~LayerNormalization() {
	Matrix::deallocateMatrix(mean, 1, size);
	Matrix::deallocateMatrix(variance, 1, size);
	Matrix::deallocateMatrix(std, 1, size);
	Matrix::deallocateMatrix(neurons, batchSize, size + 1);
	Matrix::deallocateMatrix(neuronsTranspose, size + 1, batchSize);
	Matrix::deallocateMatrix(neuronGradient, batchSize, size + 1);
}

void LayerNormalization::predict() {
	for (int j = 0; j < size; j++) {
		mean[0][j] = 0;
		for (int i = 0; i < batchSize; i++) {
			mean[0][j] += prevLayer->neurons[i][j];
		}
		mean[0][j] /= batchSize;
		variance[0][j] = 0;
		for (int i = 0; i < batchSize; i++) {
			variance[0][j] += (prevLayer->neurons[i][j] - mean[0][j]) * (prevLayer->neurons[i][j] - mean[0][j]);
		}
		variance[0][j] /= batchSize;
		std[0][j] = sqrt(variance[0][j] + 0.0000001);

		for (int i = 0; i < batchSize; i++) {
			neurons[i][j] = (prevLayer->neurons[i][j] - mean[0][j]) / std[0][j];
		}
	}
	Matrix::transpose(batchSize, size + 1, neurons, neuronsTranspose);
	if (nextLayer != NULL) {
		nextLayer->predict();
	}
}

void LayerNormalization::forwardPropagate() {
	for (int j = 0; j < size; j++) {
		mean[0][j] = 0;
		for (int i = 0; i < batchSize; i++) {
			mean[0][j] += prevLayer->neurons[i][j];
		}
		mean[0][j] /= batchSize;
		variance[0][j] = 0;
		for (int i = 0; i < batchSize; i++) {
			variance[0][j] += (prevLayer->neurons[i][j] - mean[0][j]) * (prevLayer->neurons[i][j] - mean[0][j]);
		}
		variance[0][j] /= batchSize;
		std[0][j] = sqrt(variance[0][j] + 0.0000001);

		for (int i = 0; i < batchSize; i++) {
			neurons[i][j] = (prevLayer->neurons[i][j] - mean[0][j]) / std[0][j];
		}
	}
	Matrix::transpose(batchSize, size + 1, neurons, neuronsTranspose);
	if (nextLayer != NULL) {
		nextLayer->forwardPropagate();
	}
}

void LayerNormalization::backPropagate() {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			float grad = std[0][j] * (1 - 1.0 / batchSize);
			grad -= (prevLayer->neurons[i][j] - mean[0][j]) * (prevLayer->neurons[i][j] - mean[0][j]) * (1 - 1.0 / batchSize) / (batchSize * std[0][j]);
			prevLayer->neuronGradient[i][j] = neuronGradient[i][j] * grad / variance[0][j];
		}
	}
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void LayerNormalization::setPrevLayer(Layer* prevLayer) {
	this->prevLayer = prevLayer;
	size = prevLayer->size;
	mean = Matrix::allocateMatrix(Matrix::ZERO_FILL, 1, size);
	variance = Matrix::allocateMatrix(Matrix::ZERO_FILL, 1, size);
	std = Matrix::allocateMatrix(Matrix::ZERO_FILL, 1, size);
}

void LayerNormalization::save(ofstream& file) {
	file << "LayerNormalization,\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}
