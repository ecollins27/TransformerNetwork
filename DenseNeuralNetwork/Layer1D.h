#pragma once
#include "Layer.h"
#include "Matrix.h"

class Layer1D : public Layer {

public:
	Matrix neurons;
	Matrix neuronGradient;

	virtual void setBatchSize(int batchSize) {
		this->batchSize = batchSize;
		neurons = Matrix(Matrix::ZERO_FILL, batchSize, size + 1, true);
		neuronGradient = Matrix(Matrix::ZERO_FILL, batchSize, size + 1, false);
		for (int i = 0; i < batchSize; i++) {
			neurons.r(i, size) = 1;
		}
	}
};

