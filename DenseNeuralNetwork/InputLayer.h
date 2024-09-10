#pragma once
#include "Layer.h"

class InputLayer : public Layer {

public:

	InputLayer(int size);

	void setInput(double* data);

	~InputLayer();

	void setPrevLayer(Layer* layer);
	void setNextLayer(Layer* layer);

	void forwardPropagate();
	void backPropagate();
	void applyGradients(TrainingParams* params, int t);
	void save(ofstream& file);
};

