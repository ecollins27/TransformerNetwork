#pragma once
#include "Layer1D.h"

class ResidualSave1D : public Layer1D {

public:
	Layer1D* prevLayer;

	~ResidualSave1D();

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);

	void backPropagateWithResidual(int num);
};

