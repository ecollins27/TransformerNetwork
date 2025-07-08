#include "Input1D.h"

Input1D::Input1D(int size) {
	index = 0;
	this->size = size;
}

Input1D::~Input1D() {
	Layer1D::~Layer1D();
}

void Input1D::setInput(float** input) {
	neurons.copy(input);
}

void Input1D::setSparseInput(int* input) {
	neurons.constantFill(0);
	for (int i = 0; i < batchSize; i++) {
		neurons(i, input[i]) = 1;
	}
	neurons.copyToDevice();
}

void Input1D::propagateLayer(int num) {
	return;
}

void Input1D::backPropagate(int num) {
	return;
}

void Input1D::setPrevLayer(Layer* prevLayer) {
	throw invalid_argument("Input1D cannot have previous layer");
}

void Input1D::setBatchSize(int batchSize) {
	Layer1D::setBatchSize(batchSize);
	neurons.allocateHost();
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void Input1D::save(ofstream& file) {
	file << size << "\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}