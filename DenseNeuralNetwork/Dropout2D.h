#pragma once
#include "Layer2D.h"

class Dropout2D : public Layer2D {

public:
	Layer2D* prevLayer;

	float dropoutRate;
	bool*** dropped;

	Dropout2D(float dropoutRate);

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);

	void predict(int num);
};

