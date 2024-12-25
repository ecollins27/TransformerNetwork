#pragma once
#include "Layer1D.h"

class Gated1D : public Layer1D {

	// A1 = X x W1t
	// A2 = X x W2t
	// Ao = activation(A1)
	// Y = A1 * A2t

public:
	Layer1D* prevLayer;

	Matrix2 weights1;
	Matrix2 weightGradient1;
	Matrix2 weights2;
	Matrix2 weightGradient2;

	Matrix2 A1;
	Matrix2 A1Grad;
	Matrix2 A2;
	Matrix2 A2Grad;
	Matrix2 Ao;
	Matrix2 AoGrad;

	Matrix3D activationGradient;
	Matrix2 activationGradientMatrix;

	Optimizer* optimizer1;
	Optimizer* optimizer2;

	Activation* activation;

	Gated1D(Activation* activation, int size);

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);

	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);
	int getNumParameters();

};

