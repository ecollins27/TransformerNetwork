#pragma once
#include "Layer1D.h"
#include "Layer2D.h"

class SequenceMean : public Layer1D {

public:
	Layer2D* prevLayer;

	int forwardThreadCount, backThreadCount;

	Activation* activation;

	Matrix means;
	Matrix backPropIntermediate;

	SequenceMean(Activation* activation);

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
};

