#pragma once
#include "Layer1D.h"
#include "ResidualSave1D.h"

class ResidualAdd1D : public Layer1D {

public:
	Layer1D* prevLayer;

	ResidualSave1D* residual;

	ResidualAdd1D(ResidualSave1D* residualLayer);

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
};

