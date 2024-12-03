#pragma once
#include "Layer.h"
#include "Activation.h"

class BatchMean : public Layer {

public:
	float** mean;
	float** backPropIntermediate;
	float*** activationGradient;
	bool isDiagonal;
	int prevBatchSize;
	Activation* activation;

	BatchMean(Activation* activation);

	void propagateLayer();
	void backPropagate();
	void setBatchSize(int batchSize);
	void setMaxBatchSize(int maxBatchSize);

	void setPrevLayer(Layer* prevLayer);
	void save(ofstream& file);
};

