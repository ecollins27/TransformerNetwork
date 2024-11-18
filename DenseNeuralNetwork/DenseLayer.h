#pragma once
#include "Layer.h"
#include "Activation.h"
#include "Optimizer.h"

class DenseLayer : public Layer {

public:
	float** activations;
	float** weights;
	float** weightsTranspose;
	float** weightGradient;
	float** backPropIntermediate;
	float** backPropIntermediateTranspose;

	Activation* activation;
	float*** activationGradient;
	bool isDiagonal;

	DenseLayer(Activation* activation, int size);

	~DenseLayer();

	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void setMaxBatchSize(int maxBatchSize);

	void predict();
	void forwardPropagate();
	void backPropagate();
	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);
	void save(ofstream& file);
	int getNumParameters();
};

