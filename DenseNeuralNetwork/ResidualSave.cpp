#include "ResidualSave.h"

void ResidualSave::predict() {
	Matrix::copy(batchSize, size, prevLayer->neurons, neurons);
	if (nextLayer != NULL) {
		nextLayer->predict();
	}
}

void ResidualSave::forwardPropagate() {
	Matrix::copy(batchSize, size, prevLayer->neurons, neurons);
	Matrix::transpose(batchSize, size, neurons, neuronsTranspose);
	if (nextLayer != NULL) {
		nextLayer->forwardPropagate();
	}
}

void ResidualSave::backPropagate() {
	return;
}

void ResidualSave::backPropagateWithResidual() {
	Matrix::copy(batchSize, size, neuronGradient, prevLayer->neuronGradient);
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void ResidualSave::setPrevLayer(Layer* prevLayer) {
	this->prevLayer = prevLayer;
	this->prevSize = prevLayer->size;
	this->size = prevSize;
}

void ResidualSave::setNextLayer(Layer* nextLayer) {
	this->nextLayer = nextLayer;
}

void ResidualSave::setBatchSize(int batchSize) {
	if (neurons != NULL) {
		Matrix::deallocateMatrix(neurons, batchSize, size + 1);
		Matrix::deallocateMatrix(neuronsTranspose, size + 1, batchSize);
		Matrix::deallocateMatrix(neuronGradient, batchSize, size + 1);
	}
	this->batchSize = batchSize;
	neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
	neuronsTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, batchSize);
	neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
	for (int i = 0; i < batchSize; i++) {
		neurons[i][size] = 1;
		neuronsTranspose[size][i] = 1;
	}
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void ResidualSave::applyGradients(float learningRate, int t) {
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void ResidualSave::setOptimizer(Optimizer* optimizer) {
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

void ResidualSave::save(ofstream& file) {
	file << "ResidualSave,\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}