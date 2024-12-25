#pragma once
#include "Layer1D.h"

class Dense1D : public Layer1D {

public:
	Layer1D* prevLayer;

	Matrix2 weights;
	Matrix2 weightGradient;
	Matrix2 linearCombo;
	Matrix3D activationGradient;
	Matrix2 activationGradientMatrix;
	Matrix2 backPropIntermediate;

	Activation* activation;
	Optimizer* optimizer;

	Dense1D(Activation* activation, int size);

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);

	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);
	int getNumParameters();
};

