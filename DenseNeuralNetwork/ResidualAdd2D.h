#pragma once
#include "Layer2D.h"
#include "ResidualSave2D.h"

class ResidualAdd2D : public Layer2D {

public:
	Layer2D* prevLayer;
	ResidualSave2D* residual;

	ResidualAdd2D(ResidualSave2D* residualLayer);

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void save(ofstream& file);
};

