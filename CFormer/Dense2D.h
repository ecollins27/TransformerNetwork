#pragma once
#include "Layer2D.h"

class Dense2D : public Layer2D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer = NULL;

	Matrix weights;
	Matrix weightGradient;
	Matrix* linearCombo;
	Matrix* backPropIntermediate;

	Activation* activation;
	Optimizer<Matrix>* optimizer;

	Dense2D(Activation* activation, int size);

	void initPropagationQueue(OperationQueue& queue);
	void initBackPropQueue(OperationQueue& queue);
	void initApplicationQueue(OperationQueue& queue, float learningRate, int& t);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);

	void setNumTokens(int* numTokens);
	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer<>* optimizer);
	int getNumParameters();
};

