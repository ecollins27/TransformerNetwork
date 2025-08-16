#pragma once
#include "Layer1D.h"

class ResidualSave1D : public Layer1D {

public:
	const static string LAYER_NAME;

	Layer1D* prevLayer = NULL;

	~ResidualSave1D();

	void initPropagationQueue(OperationQueue& queue);
	void initBackPropQueue(OperationQueue& queue);
	void initBackPropQueueWithResidual(OperationQueue& queue);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);
};

