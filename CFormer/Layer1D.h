#pragma once
#include "Layer.h"
#include "Matrix.h"
#include "FillFunction.h"

class Layer1D : public Layer {

public:
	Matrix neurons;
	Matrix neuronGradient;

	~Layer1D();

	virtual void setBatchSize(const int batchSize) {
		this->batchSize = batchSize;
		neurons = Matrix(batchSize, size + 1);
		neurons.setLayerOutput(true);
		neuronGradient = Matrix(batchSize, size + 1);
	}
};

