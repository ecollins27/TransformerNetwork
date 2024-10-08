#pragma once
#include "Layer.h"

class Dropout : public Layer {

public:
	float dropRate, scale;
	bool** dropped;

	Dropout(float dropRate);
	~Dropout();

	void setPrevLayer(Layer* layer);
	void setNextLayer(Layer* layer);
	void setBatchSize(int batchSize);

	void predict();
	void forwardPropagate();
	void backPropagate();
	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);
	void save(ofstream& file);
};

