#include "Input2D.h"

Input2D::Input2D(int size) {
	index = 0;
	this->size = size;
}

void Input2D::setInput(float*** input) {
	for (int n = 0; n < batchSize; n++) {
		for (int i = 0; i < numTokens[n]; i++) {
			for (int j = 0; j < size; j++) {
				neurons[n](i, j) = input[n][i][j];
			}
		}
	}
}

void Input2D::setSparseInput(int** input) {
	for (int n = 0; n < batchSize; n++) {
		neurons[n].fill(FillFunction::ZERO_FILL);
		for (int i = 0; i < numTokens[n]; i++) {
			neurons[n](i, input[n][i]) = 1;
			neurons[n](i, size) = 1;
		}
	}
}

void Input2D::initPropagationQueue(OperationQueue& queue) {
	return;
}

void Input2D::initBackPropQueue(OperationQueue& queue) {
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