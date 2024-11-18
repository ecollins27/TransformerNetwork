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

	void save(ofstream& file);

};

