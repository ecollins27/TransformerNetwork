#pragma once
#include "Layer1D.h"

class Dense1D : public Layer1D {

public:
	const static string LAYER_NAME;

	Layer1D* prevLayer = NULL;

	Matrix weights;
	Matrix weightGradient;
	Matrix linearCombo;
	Matrix backPropIntermediate;

	Activation* activation;
	Optimizer<Matrix>* optimizer;

	Dense1D(Activation* activation, int size);
	~Dense1D();

	void initPropagationQueue(OperationQueue& queue);
	void initBackPropQueue(OperationQueue& queue);
	void initApplicationQueue(OperationQueue& queue, float learningRate, int& t);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);

	void setOptimizer(Optimizer<>* optimizer);
	int getNumParameters();
};

