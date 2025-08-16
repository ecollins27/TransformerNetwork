#pragma once
#include "TrainingParams.h"
#include "Optimizer.h"
#include "Activation.h"
#include "OperationQueue.h"
#include <fstream>

class InputLayer;
class Model;
class ModelParser;

class Layer {

public:
	// does not include bias
	int size;
	//includes bias
	int prevSize;
	int batchSize;
	int index;

	Layer* nextLayer = NULL;

	~Layer();

	template<typename T, typename A>
	static bool instanceOf(A l) {
		return dynamic_cast<T*>(l) != NULL;
	}

	virtual void initPropagationQueue(OperationQueue& queue) = 0;
	virtual void initBackPropQueue(OperationQueue& queue) = 0;
	virtual void setPrevLayer(Layer* prevLayer) = 0;
	virtual void setBatchSize(int batchSize) = 0;
	virtual void save(ofstream& file) = 0;

	virtual void initApplicationQueue(OperationQueue& queue, float learningRate, int& t) {
		if (nextLayer != NULL) {
			nextLayer->initApplicationQueue(queue, learningRate, t);
		}
	}
	virtual void initForwardPropQueue(OperationQueue& queue) {
		initPropagationQueue(queue);
		if (nextLayer != NULL) {
			nextLayer->initForwardPropQueue(queue);
		}
	}

	virtual void initPredictQueue(OperationQueue& queue) {
		initPropagationQueue(queue);
		if (nextLayer != NULL) {
			nextLayer->initPredictQueue(queue);
		}
	}

	virtual void setNextLayer(Layer* nextLayer) {
		this->nextLayer = nextLayer;
	}
	virtual void setOptimizer(Optimizer<>* optimizer) {
		if (nextLayer != NULL) {
			nextLayer->setOptimizer(optimizer);
		}
	}
	virtual int getNumParameters() {
		if (nextLayer != NULL) {
			return nextLayer->getNumParameters();
		}
		return 0;
	};
	virtual void summary() {
		string className = typeid(*this).name();
		className = className.substr(6, className.length());
		printf("%s", className.c_str());
		for (int i = className.length(); i < 30; i++) {
			printf(" ");
		}
		if (nextLayer != NULL) {
			printf("NumParameters:%d\n", getNumParameters() - nextLayer->getNumParameters());
		}
		else {
			printf("NumParameters:%d\n", getNumParameters());
		}
		if (nextLayer != NULL) {
			nextLayer->summary();
		}
	}
};

