#include "Input2D.h"

Input2D::Input2D(int size) {
	index = 0;
	this->size = size;
}

void Input2D::setInput(int num, float** input) {
	neurons[num].copy(numTokens[num], size, input);
}

void Input2D::setSparseInput(int num, int* input) {
	neurons[num].constantFill(0);
	for (int i = 0; i < numTokens[num]; i++) {
		neurons[num](i, input[i]) = 1;
		neurons[num](i, size) = 1;
	}
}

void Input2D::propagateLayer(int num) {
	return;
}

void Input2D::backPropagate(int num) {
	return;
}

void Input2D::setPrevLayer(Layer* prevLayer) {
	throw invalid_argument("Input1D cannot have previous layer");
}

void Input2D::setBatchSize(int batchSize) {
	Layer2D::initNeurons(batchSize);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void Input2D::save(ofstream& file) {
	file << size << "\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}