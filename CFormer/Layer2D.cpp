#include "Layer2D.h"

void Layer2D::initNeurons(int batchSize) {
	this->batchSize = batchSize;
	neurons = Matrix::allocateMatrixArray(batchSize, maxNumTokens, size + 1);
	for (int i = 0; i < batchSize; i++) {
		neurons[i].setLayerOutput(true);
	}
	neuronGradient = Matrix::allocateMatrixArray(batchSize, maxNumTokens, size + 1);
}

void Layer2D::updateNeuronDimensions() {
	for (int i = 0; i < batchSize; i++) {
		neurons[i].setHeight(numTokens[i]);
		neuronGradient[i].setHeight(numTokens[i]);
	}
}