#pragma once
#include "Layer2D.h"

class PositionalEncoding2D : public Layer2D {

public:
	Layer2D* prevLayer;
	float L;

	PositionalEncoding2D(float L = 10000);

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void save(ofstream& file);
};

