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

void InputLayer::save(ofstream& file) {
	file << size << "\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}