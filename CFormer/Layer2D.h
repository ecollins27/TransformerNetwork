#pragma once
#include "Layer.h"
#include "Matrix2.h"

class Layer2D : public Layer {

public:
	int maxNumTokens;
	int* numTokens;

	Matrix2* neurons;
	Matrix2* neuronGradient;

	virtual void setNumTokens(int* numTokens) {
		this->numTokens = numTokens;
		updateNeuronDimensions();
		if (nextLayer != NULL && instanceOf<Layer2D>(nextLayer)){
			((Layer2D*)nextLayer)->setNumTokens(numTokens);
		}
	}

	virtual void setMaxNumTokens(int maxNumTokens) {
		this->maxNumTokens = maxNumTokens;
		if (nextLayer != NULL && instanceOf<Layer2D>(nextLayer)) {
			((Layer2D*)nextLayer)->setMaxNumTokens(maxNumTokens);
		}
	}

	void initNeurons(int batchSize);
	void updateNeuronDimensions();

	virtual void setBatchSize(int batchSize) {
		initNeurons(batchSize);
		if (nextLayer != NULL) {
			nextLayer->setBatchSize(batchSize);
		}
	}
};

