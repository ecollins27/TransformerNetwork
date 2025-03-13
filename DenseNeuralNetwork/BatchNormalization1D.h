#pragma once
#include "Layer1D.h"
#include "Activation.h"

class BatchNormalization1D : public Layer1D {

public:
	Layer1D* prevLayer;

	float momentum;
	Matrix mean, batchMean;
	Matrix variance, batchVariance;
	Matrix std;
	Matrix parameters;
	Matrix parameterGradient;

	Optimizer* optimizer;

	BatchNormalization1D(float momentum);
	~BatchNormalization1D();

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);

	void predict(int num);
	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);
	int getNumParameters();
};

