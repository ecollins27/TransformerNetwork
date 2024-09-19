#pragma once
#include "Layer.h"

class Dropout : public Layer {

public:
	double dropRate, scale;
	bool** dropped;

	Dropout(double dropRate);
	~Dropout();

	void setPrevLayer(Layer* layer);
	void setNextLayer(Layer* layer);
	void setBatchSize(int batchSize);

	void predict();
	void forwardPropagate();
	void backPropagate();
	void applyGradients(TrainingParams* params, int t);
	void setOptimizer(Optimizer* optimizer);
	void save(ofstream& file);
};

