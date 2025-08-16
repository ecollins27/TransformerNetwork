#pragma once
#include "Layer2D.h"

class Dropout2D : public Layer2D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer = NULL;

	float dropoutRate;
	bool*** dropped;

	uniform_real_distribution<float> distribution;
	default_random_engine generator;

	Dropout2D(float dropoutRate);

	void initPropagationQueue(OperationQueue& queue);
	void initBackPropQueue(OperationQueue& queue);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);

	void initPredictQueue(OperationQueue& queue);
};

