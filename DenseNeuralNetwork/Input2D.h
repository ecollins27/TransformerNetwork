#pragma once
#include "Layer2D.h"

class Input2D : public Layer2D {

public:
	Input2D(int size);

	void setInput(int num, float** input);
	void setSparseInput(int num, int* input);
	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void save(ofstream& file);
};

