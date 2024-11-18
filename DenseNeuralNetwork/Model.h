#pragma once
#include "Loss.h"
#include "DenseLayer.h"
#include "Dropout.h"
#include <fstream>

class Model {

public:
	InputLayer* inputLayer;
	Layer* outputLayer;
	int t;

	virtual void addLayer(Layer* layer) {
		layer->setPrevLayer(outputLayer);
		outputLayer->setNextLayer(layer);
		outputLayer = layer;
	}

	template<typename T>
	T* getLayer(int index) {
		Layer* layer = inputLayer;
		for (int i = 0; i < index; i++) {
			layer = layer->nextLayer;
		}
		return (T*)layer;
	}
	virtual void applyGradients(float learningRate) {
		t++;
		inputLayer->applyGradients(learningRate, t);
	}
	virtual void setTrainable(bool trainable) {
		inputLayer->setTrainable(trainable);
	}
	virtual void save(string fileName) = 0;
	virtual int getNumParameters() {
		return inputLayer->getNumParameters();
	}
};

