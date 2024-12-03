#include "LayerNormalization.h"

LayerNormalization::~LayerNormalization() {
	Matrix::deallocateMatrix(mean, 1, size);
	Matrix::deallocateMatrix(variance, 1, size);
	Matrix::deallocateMatrix(std, 1, size);
	Matrix::deallocateMatrix(neurons, batchSize, size + 1);
	Matrix::deallocateMatrix(neuronsTranspose, size + 1, batchSize);
	Matrix::deallocateMatrix(neuronGradient, batchSize, size + 1);
}

void LayerNormalization::propagateLayer() {
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
		std[0][j] = sqrt(variance[0][j]);

		for (int i = 0; i < batchSize; i++) {
			if (std[0][j] == 0) {
				neurons[i][j] = 0;
			}
			else {
				neurons[i][j] = (prevLayer->neurons[i][j] - mean[0][j]) / std[0][j];
			}
		}
	}
	Matrix::transpose(batchSize, size, neurons, neuronsTranspose);
}

void LayerNormalization::backPropagate() {
	float grad = 0;
	float inverseBatchSize = 1.0 / batchSize;
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			prevLayer->neuronGradient[i][j] = 0;
			for (int k = 0; k < batchSize; k++) {
				grad = ((k == i ? 1 : 0) - inverseBatchSize) - inverseBatchSize * neurons[k][j] * neurons[i][j];
				grad /= std[0][j];
				if (variance[0][j] != 0) {
					prevLayer->neuronGradient[i][j] += neuronGradient[k][j] * grad;
				}
			}
		}
	}
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void LayerNormalization::setPrevLayer(Layer* prevLayer) {
	this->prevLayer = prevLayer;
	size = prevLayer->size;
	prevSize = size + 1;
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
