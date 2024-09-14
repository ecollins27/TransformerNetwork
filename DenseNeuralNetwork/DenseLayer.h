#pragma once
#include "Layer.h"
#include "Activation.h"
#include "Optimizer.h"

class DenseLayer : public Layer {

public:
	double** activations;
	double** weights;
	double** weightGradient;
	double** weightM1;
	double** weightM2;
	double** weightS;
	double** backPropIntermediate;

	Activation* activation;
	double*** activationGradient;
	bool isDiagonal;

	DenseLayer(Activation* activation, int size);

	~DenseLayer();

	void setPrevLayer(Layer* prevLayer);
	void setNextLayer(Layer* nextLayer);
	void setBatchSize(int batchSize);

	void forwardPropagate();
	void backPropagate();
	void applyGradients(TrainingParams* params, int t);
	void save(ofstream& file);
};

