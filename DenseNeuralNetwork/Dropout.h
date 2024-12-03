#pragma once
#include "Layer.h"

class Dropout : public Layer {

public:
	float dropRate, scale;
	bool** dropped;

	Dropout(float dropRate);
	~Dropout();

	void setPrevLayer(Layer* layer);
	void setBatchSize(int batchSize);
	void setMaxBatchSize(int maxBatchSize);

	void propagateLayer() { return; }
	void predict();
	void forwardPropagate();
	void backPropagate();
	void save(ofstream& file);
};

