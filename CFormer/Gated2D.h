#pragma once
#include "Layer2D.h"

class Gated2D : public Layer2D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer = NULL;

	Matrix2 weights1;
	Matrix2* weightGradient1;
	Matrix2 weights2;
	Matrix2* weightGradient2;

	Matrix2* A1;
	Matrix2* A1Grad;
	Matrix2* A2;
	Matrix2* A2Grad;
	Matrix2* Ao;
	Matrix2* AoGrad;

	Optimizer* optimizer1;
	Optimizer* optimizer2;

	Activation* activation;

	Gated2D(Activation* activation, int size);

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);

	void setNumTokens(int* numTokens);
	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);
	int getNumParameters();
};

