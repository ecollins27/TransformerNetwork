#include "BatchMean.h"

BatchMean::BatchMean(Activation* activation) {
	this->activation = activation->clone();
	isDiagonal = activation->isDiagonal();
}

void BatchMean::propagateLayer() {
	for (int i = 0; i < prevBatchSize; i++) {
		for (int j = 0; j < size; j++) {
			if (i == 0) {
				mean[0][j] = prevLayer->neurons[i][j];
			}
			else {
				mean[0][j] += prevLayer->neurons[i][j];
			}
		}
	}
	Matrix::scale(1, size, mean, 1.0 / batchSize);
	activation->operate(1, size, mean, neurons);
	Matrix::transpose(1, size, neurons, neuronsTranspose);
}

void BatchMean::backPropagate() {
	activation->differentiate(1, size, mean, neurons, activationGradient);
	if (isDiagonal) {
		Matrix::elementMultiply(1, size, neuronGradient, activationGradient[0], backPropIntermediate, true);
	}
	else {
		Matrix::matrixTensorMultiply(1, size, size, neuronGradient, activationGradient, backPropIntermediate, true);
	}
	for (int i = 0; i < prevBatchSize; i++) {
		Matrix::copy(1, size, backPropIntermediate, &prevLayer->neuronGradient[i]);
	}
	Matrix::scale(batchSize, size, prevLayer->neuronGradient, 1.0 / batchSize);
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void BatchMean::setBatchSize(int batchSize) {
	this->batchSize = 1;
	this->prevBatchSize = batchSize;
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(1);
	}
}

void BatchMean::setMaxBatchSize(int maxBatchSize) {
	this->maxBatchSize = maxBatchSize;
	if (nextLayer != NULL) {
		nextLayer->setMaxBatchSize(1);
	}
}

void BatchMean::setPrevLayer(Layer* prevLayer) {
	this->prevLayer = prevLayer;
	this->prevSize = prevLayer->size;
	size = prevSize;
	mean = Matrix::allocateMatrix(Matrix::ZERO_FILL, 1, size);
	backPropIntermediate = Matrix::allocateMatrix(Matrix::ZERO_FILL, 1, size);
	neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, 1, size + 1);
	neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, 1, size + 1);
	neuronsTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, 1);
	if (isDiagonal) {
		activationGradient = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, 1, 1, size);
	}
	else {
		activationGradient = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, 1, size, size);
	}
	neurons[0][size] = 1;
	neuronsTranspose[size][0] = 1;
}

void BatchMean::save(ofstream& file) {
	file << "BatchMean,";
	activation->save(file);
	file << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}