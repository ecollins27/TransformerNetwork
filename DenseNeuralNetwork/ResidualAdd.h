#pragma once
#include "Layer.h"
#include "ResidualSave.h"

class ResidualAdd : public Layer {

public:
	ResidualSave* saveLayer;

	ResidualAdd(ResidualSave* saveLayer);
	void predict();
	void forwardPropagate();
	void backPropagate();

	void setPrevLayer(Layer* prevLayer);
	void setNextLayer(Layer* nextLayer);
	void setBatchSize(int batchSize);
	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);

	void save(ofstream& file);

};

