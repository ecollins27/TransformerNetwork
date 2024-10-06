#pragma once
#include "Layer.h"
#include "Activation.h"


class GatedLayer : public Layer {

public:
	double** weights1;
	double** weights2;
	double** weightGradient1;
	double** weightGradient2;
	double** activations1;
	double** activations2;
	double** activationOutput;
	double*** activationGradient;
	double** backPropIntermediate1;
	double** backPropIntermediate2;
	double** backPropIntermediate3;

	Activation* activation;
	bool isDiagonal;

	Optimizer* optimizer2;

	GatedLayer(Activation* activation, int size);

	void predict();
	void forwardPropagate();
	void backPropagate();

	void setPrevLayer(Layer* prevLayer);
	void setNextLayer(Layer* nextLayer);
	void setBatchSize(int batchSize);
	void applyGradients(double learningRate, int t);
	void setOptimizer(Optimizer* optimizer);

	void save(ofstream& file);
	int getNumParameters();
};

