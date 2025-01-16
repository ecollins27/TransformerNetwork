#pragma once
#include "TrainingParams.h"
#include "Optimizer.h"
#include "Activation.h"
#include "Matrix3D.h"
#include <fstream>

class InputLayer;

class Layer {

public:
	// does not include bias
	int size;
	//includes bias
	int prevSize;
	int batchSize;
	int index;

	Layer* nextLayer;

	template<typename T, typename A>
	static bool instanceOf(A l) {
		return dynamic_cast<T*>(l) != NULL;
	}

	virtual void propagateLayer(int num) = 0;
	virtual void backPropagate(int num) = 0;
	virtual void setPrevLayer(Layer* prevLayer) = 0;
	virtual void setBatchSize(int batchSize) = 0;
	virtual void save(ofstream& file) = 0;

	virtual void predict(int num) {
		propagateLayer(num);
		if (nextLayer != NULL) {
			nextLayer->predict(num);
		}
	}
	virtual void forwardPropagate(int num) {
		propagateLayer(num);
		if (nextLayer != NULL) {
			nextLayer->forwardPropagate(num);
		}
	}

	virtual void setNextLayer(Layer* nextLayer) {
		this->nextLayer = nextLayer;
	}
	virtual void applyGradients(float learningRate, int t) {
		if (nextLayer != NULL) {
			nextLayer->applyGradients(learningRate, t);
		}
	}
	virtual void setOptimizer(Optimizer* optimizer) {
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

