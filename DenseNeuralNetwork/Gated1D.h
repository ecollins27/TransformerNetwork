#pragma once
#include "Layer1D.h"

class Gated1D : public Layer1D {

	// A1 = X x W1t
	// A2 = X x W2t
	// Ao = activation(A1)
	// Y = A1 * A2t

public:
	Layer1D* prevLayer;

	Matrix weights1;
	Matrix weightGradient1;
	Matrix weights2;
	Matrix weightGradient2;

	Matrix A1;
	Matrix A1Grad;
	Matrix A2;
	Matrix A2Grad;
	Matrix Ao;
	Matrix AoGrad;

	Optimizer* optimizer1;
	Optimizer* optimizer2;

	Activation* activation;

	Gated1D(Activation* activation, int size);
	~Gated1D();

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);

	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);
	int getNumParameters();

};

