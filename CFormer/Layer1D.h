#pragma once
#include "Layer.h"
#include "Matrix2.h"
#include "FillFunction.h"

class Layer1D : public Layer {

public:
	Matrix2 neurons;
	Matrix2 neuronGradient;

	~Layer1D();

	virtual void setBatchSize(int batchSize) {
		this->batchSize = batchSize;
		neurons = Matrix2(batchSize, size + 1, 0);
		neurons.constantFill(1);
		neuronGradient = Matrix2(batchSize, size, 0);
	}
};

