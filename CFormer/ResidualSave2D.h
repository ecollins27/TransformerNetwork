#pragma once
#include "Layer2D.h"

class ResidualSave2D : public Layer2D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer = NULL;

	void initPropagationQueue(OperationQueue& queue);
	void initBackPropQueue(OperationQueue& queue);
	void initBackPropQueueWithResidual(OperationQueue& queue);
	void setPrevLayer(Layer* prevLayer);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);

};