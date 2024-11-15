#include "ResidualAdd.h"

ResidualAdd::ResidualAdd(ResidualSave* saveLayer) {
	this->saveLayer = saveLayer;
}

void ResidualAdd::predict() {
	Matrix::copy(batchSize, size, prevLayer->neurons, neurons);
	Matrix::add(batchSize, size, neurons, saveLayer->neurons, neurons, 1, 1);
	if (nextLayer != NULL) {
		nextLayer->predict();
	}
}

void ResidualAdd::forwardPropagate() {
	Matrix::copy(batchSize, size, prevLayer->neurons, neurons);
	Matrix::add(batchSize, size, neurons, saveLayer->neurons, neurons, 1, 1);
	Matrix::transpose(batchSize, size, neurons, neuronsTranspose);
	if (nextLayer != NULL) {
		nextLayer->predict();
	}
}

void ResidualAdd::backPropagate() {
	Matrix::copy(batchSize, size, neuronGradient, prevLayer->neuronGradient);
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
	Matrix::add(batchSize, size, neuronGradient, saveLayer->neuronGradient, saveLayer->neuronGradient, 1, 1);
	saveLayer->backPropagateWithResidual();
}

void ResidualAdd::setPrevLayer(Layer* prevLayer) {
	this->prevLayer = prevLayer;
	this->prevSize = prevLayer->size;
	this->size = prevSize;
}

void ResidualAdd::setNextLayer(Layer* nextLayer) {
	this->nextLayer = nextLayer;
}

void ResidualAdd::setBatchSize(int batchSize) {
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

void ResidualAdd::applyGradients(float learningRate, int t) {
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void ResidualAdd::setOptimizer(Optimizer* optimizer) {
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

void ResidualAdd::save(ofstream& file) {
	file << "ResidualAdd,\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}