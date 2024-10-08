#include "InputLayer.h"

InputLayer::InputLayer(int size) {
	this->size = size;

	prevLayer = NULL;
	nextLayer = NULL;
}

InputLayer::~InputLayer() {
	Matrix::deallocateMatrix(neurons, batchSize, size + 1);
	Matrix::deallocateMatrix(neuronGradient, batchSize, size + 1);
	Matrix::deallocateMatrix(neuronsTranspose, size + 1, batchSize);
}

void InputLayer::setInput(float** data) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			neurons[i][j] = data[i][j];
			neuronsTranspose[j][i] = data[i][j];
		}
	}
}

void InputLayer::setPrevLayer(Layer* layer) {
	throw invalid_argument("Input Layer must be first layer");
}

void InputLayer::setNextLayer(Layer* layer) {
	nextLayer = layer;
}

void InputLayer::setBatchSize(int batchSize) {
	if (neurons != NULL) {
		Matrix::deallocateMatrix(neurons, this->batchSize, size + 1);
		Matrix::deallocateMatrix(neuronsTranspose, size + 1, this->batchSize);
		Matrix::deallocateMatrix(neuronGradient, this->batchSize, size + 1);
	}
	this->batchSize = batchSize;
	neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
	neuronsTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, batchSize);
	neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
	for (int i = 0; i < batchSize; i++) {
		neurons[i][size] = 1;
	}
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void InputLayer::predict() {
	if (nextLayer != NULL) {
		nextLayer->predict();
	}
}

void InputLayer::forwardPropagate() {
	if (nextLayer != NULL) {
		nextLayer->forwardPropagate();
	}
}

void InputLayer::backPropagate() {
	return;
}

void InputLayer::applyGradients(float learningRate, int t) {
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void InputLayer::setOptimizer(Optimizer* optimizer) {
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

void InputLayer::save(ofstream& file) {
	file << size << "\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}