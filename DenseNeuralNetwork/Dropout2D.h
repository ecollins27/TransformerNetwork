#pragma once
#include "Layer2D.h"

class Dropout2D : public Layer2D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer;

	float dropoutRate;
	bool*** dropped;

	uniform_real_distribution<float> distribution;
	default_random_engine generator;

	Dropout2D(float dropoutRate);

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);

	void predict(int num);
};

