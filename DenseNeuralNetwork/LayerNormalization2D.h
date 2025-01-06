#pragma once
#include "Layer2D.h"

class LayerNormalization2D : public Layer2D {

public:
	Layer2D* prevLayer;

	Matrix mean;
	Matrix variance;
	Matrix std;

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void save(ofstream& file);
};

