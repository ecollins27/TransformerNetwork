#include "Layer2D.h"

void Layer2D::initNeurons(int batchSize) {
	this->batchSize = batchSize;
	neurons = MatrixBatch(Matrix2::ZERO_FILL, batchSize, numTokens, size + 1, true);
	neuronGradient = MatrixBatch(Matrix2::ZERO_FILL, batchSize, numTokens, size + 1, true);
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < numTokens; j++) {
			neurons(i, j, size) = 1;
		}
	}
}