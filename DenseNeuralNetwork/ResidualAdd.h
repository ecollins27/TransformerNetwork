#pragma once
#include "Layer.h"
#include "ResidualSave.h"

class ResidualAdd : public Layer {

public:
	ResidualSave* saveLayer;

	ResidualAdd(ResidualSave* saveLayer);
	void propagateLayer();
	void backPropagate();

	void setPrevLayer(Layer* prevLayer);

	void save(ofstream& file);

};

