#pragma once
#include "Layer1D.h"
#include "Layer2D.h"

class SequenceMean : public Layer1D {

public:
	Layer2D* prevLayer;

	int forwardThreadCount, backThreadCount, numTokens;

	Activation* activation;

	Matrix2 means;
	Matrix2 backPropIntermediate;
	Matrix3D activationGradient;
	Matrix2 activationGradientMatrix;

	SequenceMean(Activation* activation);

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
};

