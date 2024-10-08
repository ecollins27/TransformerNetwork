#pragma once
#include "Layer.h"
#include "Activation.h"


class GatedLayer : public Layer {

public:
	float** weights1;
	float** weights1Transpose;
	float** weights2;
	float** weights2Transpose;
	float** weightGradient1;
	float** weightGradient2;
	float** activations1;
	float** activations2;
	float** activationOutput;
	float*** activationGradient;
	float** backPropIntermediate1;
	float** backPropIntermediate2;
	float** backPropIntermediate2Transpose;
	float** backPropIntermediate3;
	float** backPropIntermediate3Transpose;

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
	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);

	void save(ofstream& file);
	int getNumParameters();
};

