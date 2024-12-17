#pragma once
#include "Matrix.h"
#include "TrainingParams.h"
#include <fstream>

class InputLayer;

class Layer {

public:
	// does not include bias
	int size;
	//includes bias
	int prevSize;
	int batchSize;
	float** neurons;
	float** neuronsTranspose;
	float** neuronGradient;
	bool trainable = true;
	int maxBatchSize = -1;

	Optimizer* optimizer;

	Layer* prevLayer;
	Layer* nextLayer;

	virtual void propagateLayer() = 0;
	virtual void predict() {
		propagateLayer();
		if (nextLayer != NULL) {
			nextLayer->predict();
		}
	}
	virtual void forwardPropagate() {
		propagateLayer();
		if (nextLayer != NULL) {
			nextLayer->forwardPropagate();
		}
	}
	virtual void backPropagate() = 0;

	virtual void setPrevLayer(Layer* prevLayer) = 0;
	virtual void setNextLayer(Layer* nextLayer) {
		this->nextLayer = nextLayer;
	}
	virtual void setBatchSize(int batchSize) {
		if (maxBatchSize > 0) {
			this->batchSize = batchSize;
		} else {
			if (neurons != NULL) {
				Matrix::deallocateMatrix(neurons, this->batchSize, size + 1);
				Matrix::deallocateMatrix(neuronsTranspose, size + 1, this->batchSize);
				Matrix::deallocateMatrix(neuronGradient, this->batchSize, size + 1);
			}
			this->batchSize = batchSize;
			neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
			neuronsTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, batchSize);
			neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
			for (int i = 0; i < batchSize; i++) {
				neurons[i][size] = 1;
				neuronGradient[i][size] = 0;
			}
		}
		if (nextLayer != NULL) {
			nextLayer->setBatchSize(batchSize);
		}
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
	void setTrainable(bool trainable);

	virtual void save(ofstream& file) = 0;
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
	virtual void setMaxBatchSize(int maxBatchSize) {
		if (maxBatchSize > 0) {
			if (neurons != NULL && batchSize > 0) {
				Matrix::deallocateMatrix(neurons, this->maxBatchSize, size + 1);
				Matrix::deallocateMatrix(neuronGradient, this->maxBatchSize, size + 1);
				Matrix::deallocateMatrix(neuronsTranspose, size + 1, this->maxBatchSize);
			}
			this->maxBatchSize = maxBatchSize;
			neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size + 1);
			neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size + 1);
			neuronsTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, maxBatchSize);
		} else {
			this->maxBatchSize = maxBatchSize;
		}
		if (nextLayer != NULL) {
			nextLayer->setMaxBatchSize(maxBatchSize);
		}
	}
	int getLayerIndex();
};

