#pragma once
#include "Layer1D.h"

class Dropout1D : public Layer1D {

public:
	Layer1D* prevLayer;

	float dropoutRate;
	bool** dropped;

	Dropout1D(float dropoutRate);
	~Dropout1D();

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);

	void predict(int num);
};

