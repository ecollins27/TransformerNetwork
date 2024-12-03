#pragma once
#include "Layer.h"

class PositionalEncodingLayer : public Layer {

public:
	int L = 10000;

	PositionalEncodingLayer();
	PositionalEncodingLayer(int L);
	void propagateLayer();
	void backPropagate();

	void setPrevLayer(Layer* prevLayer);
	void save(ofstream& file);
};

