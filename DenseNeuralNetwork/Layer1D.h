#pragma once
#include "Layer.h"
#include "Matrix2.h"

class Layer1D : public Layer {

public:
	Matrix2 neurons;
	Matrix2 neuronGradient;

	virtual void setBatchSize(int batchSize) {
		this->batchSize = batchSize;
		neurons = Matrix2(Matrix2::ZERO_FILL, batchSize, size + 1, true);
		neuronGradient = Matrix2(Matrix2::ZERO_FILL, batchSize, size + 1, false);
		for (int i = 0; i < batchSize; i++) {
			neurons(i, size) = 1;
		}
	}
};

