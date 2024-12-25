#pragma once
#include "Layer2D.h"

class Dense2D : public Layer2D {

public:
	Layer2D* prevLayer;

	Matrix2 weights;
	MatrixBatch weightGradient;
	MatrixBatch linearCombo;
	Matrix3DBatch activationGradient;
	MatrixBatch activationGradientMatrix;
	MatrixBatch backPropIntermediate;

	Activation* activation;
	Optimizer* optimizer;

	Dense2D(Activation* activation, int size);

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);

	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);
	int getNumParameters();
};

