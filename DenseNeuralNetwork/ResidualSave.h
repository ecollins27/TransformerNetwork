#pragma once
#include "Layer.h"

class ResidualSave : public Layer {

public:
	void propagateLayer();
	void backPropagate();
	void backPropagateWithResidual();

	void setPrevLayer(Layer* prevLayer);

	void save(ofstream& file);

};

