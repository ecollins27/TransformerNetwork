#pragma once
#include "Matrix.h"
#include "TrainingParams.h"
#include <fstream>

class Layer {

public:
	// does not include bias
	int size;
	//includes bias
	int prevSize;
	int batchSize;
	float** neurons;
	float** neuronsTranspose;
	float** neuronGradient;
	bool trainable = true;

	Optimizer* optimizer;

	Layer* prevLayer;
	Layer* nextLayer;

	virtual void predict() = 0;
	virtual void forwardPropagate() = 0;
	virtual void backPropagate() = 0;

	virtual void setPrevLayer(Layer* prevLayer) = 0;
	virtual void setNextLayer(Layer* nextLayer) = 0;
	virtual void setBatchSize(int batchSize) = 0;
	virtual void applyGradients(float learningRate, int t) = 0;
	virtual void setOptimizer(Optimizer* optimizer) = 0;
	void setTrainable(bool trainable);

	virtual void save(ofstream& file) = 0;
	virtual int getNumParameters() {
		if (nextLayer != NULL) {
			return nextLayer->getNumParameters();
		}
		return 0;
	};
};

