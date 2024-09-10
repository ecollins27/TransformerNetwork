#pragma once
#include "Layer.h"
#include "Activation.h"
#include "Optimizer.h"

class DenseLayer : public Layer {

public:
	double** weights;
	double** weightGradient;
	double** weightM;
	double** weightS;
	double** backPropIntermediate;

	Activation* activation;
	double** activationGradient;
	bool isDiagonal;

	DenseLayer(Activation* activation, int size);

	~DenseLayer();

	void setPrevLayer(Layer* prevLayer);
	void setNextLayer(Layer* nextLayer);

	void forwardPropagate();
	void backPropagate();
	void applyGradients(TrainingParams* params, int t);
	void save(ofstream& file);
};

