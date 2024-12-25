#pragma once
#include "Layer.h"

class Layer2D : public Layer {

public:
	int numTokens, maxNumTokens;

	MatrixBatch neurons;
	MatrixBatch neuronGradient;

	virtual void setNumTokens(int numTokens) {
		this->numTokens = numTokens;
		if (nextLayer != NULL && instanceOf<Layer2D*>(nextLayer)){
			((Layer2D*)nextLayer)->setNumTokens(numTokens);
		}
	}

	virtual void setMaxNumTokens(int maxNumTokens) {
		this->maxNumTokens = maxNumTokens;
		if (nextLayer != NULL && instanceOf<Layer2D*>(nextLayer)) {
			((Layer2D*)nextLayer)->setMaxNumTokens(numTokens);
		}
	}

	void initNeurons(int batchSize);

	virtual void setBatchSize(int batchSize) {
		initNeurons(batchSize);
		if (nextLayer != NULL) {
			nextLayer->setBatchSize(batchSize);
		}
	}
};

