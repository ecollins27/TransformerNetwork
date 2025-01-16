#pragma once
#include "Layer1D.h"
class Input1D : public Layer1D {

public:

	Input1D(int size);

	void setInput(float** input);
	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
};

