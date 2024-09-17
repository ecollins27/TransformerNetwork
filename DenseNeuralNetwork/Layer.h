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
	double** neurons;
	double** neuronGradient;
	bool trainable = true;

	Layer* prevLayer;
	Layer* nextLayer;

	virtual void predict() = 0;
	virtual void forwardPropagate() = 0;
	virtual void backPropagate() = 0;

	virtual void setPrevLayer(Layer* prevLayer) = 0;
	virtual void setNextLayer(Layer* nextLayer) = 0;
	virtual void setBatchSize(int batchSize) = 0;
	virtual void applyGradients(TrainingParams* params, int t) = 0;
	void setTrainable(bool trainable);

	virtual void save(ofstream& file) = 0;
};

