#pragma once
#include "Layer2D.h"

class ResidualSave2D : public Layer2D {

public:
	Layer2D* prevLayer;

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void save(ofstream& file);

	void backPropagateWithResidual(int num);
};

