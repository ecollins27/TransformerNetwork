#pragma once
#include "Layer.h"

class Model {

public:
	virtual void addLayer(Layer* layer) = 0;
	virtual Layer* getLayer(int index) = 0;

	virtual void fit(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics, TrainingParams* params) = 0;
	virtual void test(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics) = 0;
	virtual void save(string fileName) = 0;
};

