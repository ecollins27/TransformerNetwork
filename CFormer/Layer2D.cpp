#include "Layer2D.h"

void Layer2D::initNeurons(int batchSize) {
	this->batchSize = batchSize;
	neurons = Matrix2::allocateMatrixArray(batchSize, maxNumTokens, size + 1, false);
	neuronGradient = Matrix2::allocateMatrixArray(batchSize, maxNumTokens, size, false);
}

void Layer2D::updateNeuronDimensions() {
	for (int i = 0; i < batchSize; i++) {
		neurons[i].setHeight(numTokens[i]);
		neurons[i].constantFill(1);
		neuronGradient[i].setHeight(numTokens[i]);
	}
}