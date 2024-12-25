#include "LayerNormalization2D.h"

void LayerNormalization2D::propagateLayer(int num) {
	for (int j = 0; j < size; j++) {
		float& meanSum = mean(0, j);
		for (int i = 0; i < numTokens; i++) {
			meanSum += prevLayer->neurons(num, i, j);
		}
		meanSum /= numTokens;

		float& varianceSum = variance(0, j);
		for (int i = 0; i < numTokens; i++) {
			varianceSum += (prevLayer->neurons(num, i, j) - meanSum) * (prevLayer->neurons(num, i, j) - meanSum);
		}
		varianceSum /= numTokens;
		std(0, j) = sqrt(variance(0, j));

		for (int i = 0; i < numTokens; i++) {
			if (std(0, j) == 0) {
				neurons(num, i, j) = 0;
			}
			else {
				neurons(num, i, j) = (prevLayer->neurons(num, i, j) - mean(0, j)) / std(0, j);
			}
		}
	}
}

void LayerNormalization2D::backPropagate(int num) {
	float inverseBatchSize = 1.0 / batchSize;
	prevLayer->neuronGradient[num].fill(Matrix2::ZERO_FILL, numTokens, size);
	for (int i = 0; i < numTokens; i++) {
		for (int j = 0; j < size; j++) {
			if (std(0, j) != 0) {
				for (int k = 0; k < numTokens; k++) {
					float grad = ((k == i ? 1 : 0) - inverseBatchSize) - inverseBatchSize * (prevLayer->neurons(num, k, j) - mean(0, j)) * (prevLayer->neurons(num, i, j) - mean(0, j)) / variance(0, j);
					grad /= std(0, j);
					prevLayer->neuronGradient(num,k, j) += neuronGradient(num, k, j) * grad;
				}
			}
		}
	}
	prevLayer->backPropagate(num);
}

void LayerNormalization2D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D*>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	size = prevLayer->size;
	prevSize = size + 1;
	mean = Matrix2(Matrix2::ZERO_FILL, 1, size, false);
	variance = Matrix2(Matrix2::ZERO_FILL, 1, size, false);
	std = Matrix2(Matrix2::ZERO_FILL, 1, size, false);
}

void LayerNormalization2D::save(ofstream& file) {
	file << "LayerNormalization,\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}