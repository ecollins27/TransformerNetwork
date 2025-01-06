#include "Layer2D.h"

void Layer2D::initNeurons(int batchSize) {
	this->batchSize = batchSize;
	neurons = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, maxNumTokens, size + 1, true);
	neuronGradient = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, maxNumTokens, size + 1, true);
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < maxNumTokens; j++) {
			neurons[i](j, size) = 1;
		}
	}
}