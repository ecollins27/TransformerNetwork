#pragma once
#include "Layer2D.h"

class Input2D : public Layer2D {

public:
	Input2D(int size);

	void setInput(float*** input);
	void setSparseInput(int** input);
	void initPropagationQueue(OperationQueue& queue);
	void initBackPropQueue(OperationQueue& queue);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
};

