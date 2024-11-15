#pragma once
#include "Layer.h"

class ResidualSave : public Layer {

public:
	void predict();
	void forwardPropagate();
	void backPropagate();
	void backPropagateWithResidual();

	void setPrevLayer(Layer* prevLayer);
	void setNextLayer(Layer* nextLayer);
	void setBatchSize(int batchSize);
	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);

	void save(ofstream& file);

};

