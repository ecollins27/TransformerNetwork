#include "BatchNormalization.h"
#include "Optimizer.h"

BatchNormalization::BatchNormalization() {
	momentum = 0.95;
}

BatchNormalization::BatchNormalization(float momentum) {
	this->momentum = momentum;
}
BatchNormalization::~BatchNormalization() {
	Matrix::deallocateMatrix(mean, 1, size);
	Matrix::deallocateMatrix(variance, 1, size);
	Matrix::deallocateMatrix(std, 1, size);
	Matrix::deallocateMatrix(batchMean, 1, size);
	Matrix::deallocateMatrix(batchVariance, 1, size);
	Matrix::deallocateMatrix(parameters, 2, size);
	Matrix::deallocateMatrix(parameterGradient, 2, size);
	Matrix::deallocateMatrix(neurons, batchSize, size + 1);
	Matrix::deallocateMatrix(neuronsTranspose, size + 1, batchSize);
	Matrix::deallocateMatrix(neuronGradient, batchSize, size + 1);
}

void BatchNormalization::predict() {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			neurons[i][j] = parameters[1][j] + parameters[0][j] * (prevLayer->neurons[i][j] - mean[0][j]) / std[0][j];
		}
	}
	if (nextLayer != NULL) {
		nextLayer->predict();
	}
}

void BatchNormalization::forwardPropagate() {
	for (int j = 0; j < size; j++) {
		batchMean[0][j] = 0;
		for (int i = 0; i < batchSize; i++) {
			batchMean[0][j] += prevLayer->neurons[i][j];
		}
		batchMean[0][j] /= batchSize;
		mean[0][j] = momentum * mean[0][j] + (1 - momentum) * batchMean[0][j];
		batchVariance[0][j] = 0;
		for (int i = 0; i < batchSize; i++) {
			batchVariance[0][j] += (prevLayer->neurons[i][j] - mean[0][j]) * (prevLayer->neurons[i][j] - mean[0][j]);
		}
		batchVariance[0][j] /= batchSize;
		variance[0][j] = momentum * variance[0][j] + (1 - momentum) * batchVariance[0][j];
		std[0][j] = sqrt(variance[0][j] + 0.0000001);

		for (int i = 0; i < batchSize; i++) {
			neurons[i][j] = parameters[1][j] + parameters[0][j] * (prevLayer->neurons[i][j] - mean[0][j]) / std[0][j];
		}
	}
	Matrix::transpose(batchSize, size + 1, neurons, neuronsTranspose);
	if (nextLayer != NULL) {
		nextLayer->forwardPropagate();
	}
}

void BatchNormalization::backPropagate() {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			parameterGradient[0][j] += neuronGradient[i][j] * (neurons[i][j] - parameters[1][j]) / (parameters[0][j] + 0.0000001);
			parameterGradient[1][j] += neuronGradient[i][j];
			float grad = std[0][j] * (1 - (1 - momentum) / batchSize);
			grad -= (prevLayer->neurons[i][j] - mean[0][j]) * (prevLayer->neurons[i][j] - mean[0][j]) * (1 - momentum) * (1 - (1 - momentum) / batchSize) / (batchSize * std[0][j]);
			prevLayer->neuronGradient[i][j] = neuronGradient[i][j] * parameters[0][j] * grad / variance[0][j];
		}
	}
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void BatchNormalization::setPrevLayer(Layer* prevLayer) {
	this->prevLayer = prevLayer;
	size = prevLayer->size;
	batchMean = Matrix::allocateMatrix(Matrix::ZERO_FILL, 1, size);
	batchVariance = Matrix::allocateMatrix(Matrix::ZERO_FILL, 1, size);
	mean = Matrix::allocateMatrix(Matrix::ZERO_FILL, 1, size);
	variance = Matrix::allocateMatrix(Matrix::ZERO_FILL, 1, size);
	std = Matrix::allocateMatrix(Matrix::ZERO_FILL, 1, size);
	parameters = Matrix::allocateMatrix(Matrix::ZERO_FILL, 2, size);
	for (int i = 0; i < size; i++) {
		parameters[0][i] = 1;
	}
	parameterGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, 2, size);
}

void BatchNormalization::applyGradients(float learningRate, int t) {
	Matrix::scale(2, size, parameterGradient, 1.0 / batchSize);
	optimizer->applyGradient(parameterGradient, parameters, t, learningRate);
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void BatchNormalization::setOptimizer(Optimizer* optimizer) {
	this->optimizer = optimizer->clone();
	this->optimizer->setDimensions(2, size);
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

void BatchNormalization::save(ofstream& file) {
	file << "BatchNormalization,\n";
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < size; j++) {
			if (i < 2) {
				file << parameters[i][j] << ",";
			}
			else if (i == 2) {
				file << mean[0][j] << ",";
			} else {
				file << variance[0][j] << ",";
			}
		}
		file << "\n";
	}
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

int BatchNormalization::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + 2 * size;
}