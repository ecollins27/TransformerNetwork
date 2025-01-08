#include "LayerNormalization2D.h"

void LayerNormalization2D::propagateLayer(int num) {
	for (int j = 0; j < size; j++) {
		float& meanSum = mean.r(num, j);
		for (int i = 0; i < numTokens[num]; i++) {
			meanSum += prevLayer->neurons[num](i, j);
		}
		meanSum /= numTokens[num];

		float& varianceSum = variance.r(num, j);
		for (int i = 0; i < numTokens[num]; i++) {
			varianceSum += (prevLayer->neurons[num](i, j) - meanSum) * (prevLayer->neurons[num](i, j) - meanSum);
		}
		varianceSum /= numTokens[num];
		std.r(num, j) = sqrt(varianceSum);

		for (int i = 0; i < numTokens[num]; i++) {
			if (std(num, j) == 0) {
				neurons[num].r(i, j) = 0;
			}
			else {
				neurons[num].r(i, j) = (prevLayer->neurons[num](i, j) - mean(num, j)) / std(num, j);
			}
		}
	}
}

void LayerNormalization2D::backPropagate(int num) {
	float inverseBatchSize = 1.0 / batchSize;
	prevLayer->neuronGradient[num].fill(Matrix::ZERO_FILL, numTokens[num], size);
	for (int i = 0; i < numTokens[num]; i++) {
		for (int j = 0; j < size; j++) {
			if (std(num, j) != 0) {
				for (int k = 0; k < numTokens[num]; k++) {
					float grad = ((k == i ? 1 : 0) - inverseBatchSize) - inverseBatchSize * (prevLayer->neurons[num](k, j) - mean(num, j)) * (prevLayer->neurons[num](i, j) - mean(num, j)) / variance(num, j);
					grad /= std(num, j);
					prevLayer->neuronGradient[num].r(i, j) += neuronGradient[num](k, j) * grad;
				}
			}
		}
	}
	prevLayer->backPropagate(num);
}

void LayerNormalization2D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	size = prevLayer->size;
	prevSize = size + 1;
	mean = Matrix(Matrix::ZERO_FILL, batchSize, size, false);
	variance = Matrix(Matrix::ZERO_FILL, batchSize, size, false);
	std = Matrix(Matrix::ZERO_FILL, batchSize, size, false);
}

void LayerNormalization2D::save(ofstream& file) {
	file << "LayerNormalization,\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}