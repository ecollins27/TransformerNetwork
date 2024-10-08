#pragma once
#include "Layer.h"

class InputLayer : public Layer {

public:

	InputLayer(int size);

	void setInput(float** data);

	~InputLayer();

	void setPrevLayer(Layer* layer);
	void setNextLayer(Layer* layer);
	void setBatchSize(int batchSize);

	void predict();
	void forwardPropagate();
	void backPropagate();
	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);
	void save(ofstream& file);
};

