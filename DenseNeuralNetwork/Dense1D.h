#pragma once
#include "Layer1D.h"

class Dense1D : public Layer1D {

public:
	Layer1D* prevLayer;

	Matrix weights;
	Matrix weightGradient;
	Matrix linearCombo;
	Matrix backPropIntermediate;

	Activation* activation;
	Optimizer* optimizer;

	Dense1D(Activation* activation, int size);
	~Dense1D();

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);

	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);
	int getNumParameters();
};

