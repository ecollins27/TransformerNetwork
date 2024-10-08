#pragma once
#include "Layer.h"

class BatchNormalization : public Layer {

public:
	float** batchMean;
	float** batchVariance;
	float** batchDifference;
	float** mean;
	float** variance;
	float** std;

	float** parameters;
	float** parameterGradient;
	float momentum;

	BatchNormalization();
	BatchNormalization(float momentum);
	~BatchNormalization();

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

