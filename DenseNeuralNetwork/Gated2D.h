#pragma once
#include "Layer2D.h"

class Gated2D : public Layer2D {

public:
	Layer2D* prevLayer;

	Matrix2 weights1;
	MatrixBatch weightGradient1;
	Matrix2 weights2;
	MatrixBatch weightGradient2;

	MatrixBatch A1;
	MatrixBatch A1Grad;
	MatrixBatch A2;
	MatrixBatch A2Grad;
	MatrixBatch Ao;
	MatrixBatch AoGrad;

	Matrix3DBatch activationGradient;
	MatrixBatch activationGradientMatrix;

	Optimizer* optimizer1;
	Optimizer* optimizer2;

	Activation* activation;

	Gated2D(Activation* activation, int size);

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);

	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);
	int getNumParameters();
};

