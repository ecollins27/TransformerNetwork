#pragma once
#include "Layer.h"
#include "Activation.h"
#include "Optimizer.h"

class DenseLayer : public Layer {

public:
	double** activations;
	double** weights;
	double** weightGradient;
	double** backPropIntermediate;

	Activation* activation;
	double*** activationGradient;
	bool isDiagonal;

	DenseLayer(Activation* activation, int size);

	~DenseLayer();

	void setPrevLayer(Layer* prevLayer);
	void setNextLayer(Layer* nextLayer);
	void setBatchSize(int batchSize);

	void predict();
	void forwardPropagate();
	void backPropagate();
	void applyGradients(TrainingParams* params, int t);
	void setOptimizer(Optimizer* optimizer);
	void save(ofstream& file);
};

