#pragma once
#include "Layer1D.h"
class Input1D : public Layer1D {

public:

	Input1D(int size);
	~Input1D();

	void setInput(float** input);
	void setSparseInput(int* input);
	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
};

