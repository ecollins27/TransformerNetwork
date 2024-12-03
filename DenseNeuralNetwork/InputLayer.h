#pragma once
#include "Layer.h"

class InputLayer : public Layer {

public:

	InputLayer(int size);

	void setInput(float** data);

	~InputLayer();

	void setPrevLayer(Layer* layer);

	void propagateLayer();
	void backPropagate();
	void save(ofstream& file);
};

