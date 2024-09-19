#pragma once
#include "Layer.h"

class BatchNormalization : public Layer {

public:
	double** batchMean;
	double** batchVariance;
	double** batchDifference;
	double** mean;
	double** variance;
	double** std;

	double** parameters;
	double** parameterGradient;
	double momentum;

	BatchNormalization();
	BatchNormalization(double momentum);
	~BatchNormalization();

	void predict();
	void forwardPropagate();
	void backPropagate();

	void setPrevLayer(Layer* prevLayer);
	void setNextLayer(Layer* nextLayer);
	void setBatchSize(int batchSize);
	void applyGradients(TrainingParams* params, int t);
	void setTrainable(bool trainable);
	void setOptimizer(Optimizer* optimizer);

	void save(ofstream& file);
};

