#pragma once
#include "Layer.h"

class Dropout : public Layer {

public:
	double dropRate, scale;
	bool** dropped;

	Dropout(double dropRate);

	void setPrevLayer(Layer* layer);
	void setNextLayer(Layer* layer);
	void setBatchSize(int batchSize);

	void predict();
	void forwardPropagate();
	void backPropagate();
	void applyGradients(TrainingParams* params, int t);
	void save(ofstream& file);
};

