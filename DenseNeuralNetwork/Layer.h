#pragma once
#include "Matrix.h"
#include "TrainingParams.h"

class Layer {

public:
	// does not include bias
	int size;
	//includes bias
	int prevSize;
	double** neurons;
	double** neuronGradient;

	Layer* prevLayer;
	Layer* nextLayer;

	virtual void forwardPropagate() = 0;
	virtual void backPropagate() = 0;

	virtual void setPrevLayer(Layer* prevLayer) = 0;
	virtual void setNextLayer(Layer* nextLayer) = 0;
	virtual void applyGradients(TrainingParams* params, int t) = 0;
};

