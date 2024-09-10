#include "InputLayer.h"

InputLayer::InputLayer(int size) {
	this->size = size;
	neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, 1);
	neurons[size][0] = 1;
	neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, 1);

	prevLayer = NULL;
	nextLayer = NULL;
}

InputLayer::~InputLayer() {
	Matrix::deallocateMatrix(neurons, size + 1, 1);
	Matrix::deallocateMatrix(neuronGradient, size + 1, 1);
}

void InputLayer::setInput(double* data) {
	for (int i = 0; i < size; i++) {
		neurons[i][0] = data[i];
	}
}

void InputLayer::setPrevLayer(Layer* layer) {
	throw invalid_argument("Input Layer must be first layer");
}

void InputLayer::setNextLayer(Layer* layer) {
	nextLayer = layer;
}

void InputLayer::forwardPropagate() {
	if (nextLayer != NULL) {
		nextLayer->forwardPropagate();
	}
}

void InputLayer::backPropagate() {
	return;
}

void InputLayer::applyGradients(TrainingParams* params, int t) {
	if (nextLayer != NULL) {
		nextLayer->applyGradients(params, t);
	}
}

void InputLayer::save(ofstream& file) {
	file << size << "\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}