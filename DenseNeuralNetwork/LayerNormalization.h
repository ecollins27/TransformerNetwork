#pragma once
#include "Layer.h"

class LayerNormalization : public Layer {

public:
	float** mean;
	float** variance;
	float** std;

	~LayerNormalization();

	void predict();
	void forwardPropagate();
	void backPropagate();

	void setPrevLayer(Layer* prevLayer);

	void save(ofstream& file);
};

