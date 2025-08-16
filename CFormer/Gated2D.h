#pragma once
#include "Layer2D.h"

class Gated2D : public Layer2D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer = NULL;

	Matrix weights1;
	Matrix weightGradient1;
	Matrix weights2;
	Matrix weightGradient2;

	Matrix* A1;
	Matrix* A1Grad;
	Matrix* A2;
	Matrix* A2Grad;
	Matrix* Ao;
	Matrix* AoGrad;

	Optimizer<Matrix>* optimizer1;
	Optimizer<Matrix>* optimizer2;

	Activation* activation;

	Gated2D(Activation* activation, int size);

	void initPropagationQueue(OperationQueue& queue);
	void initBackPropQueue(OperationQueue& queue);
	void initApplicationQueue(OperationQueue& queue, float learningRate, int& t);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);

	void setNumTokens(int* numTokens);
	void setOptimizer(Optimizer<>* optimizer);
	int getNumParameters();
};

