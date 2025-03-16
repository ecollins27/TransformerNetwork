#pragma once
#include "Layer2D.h"

class Gated2D : public Layer2D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer;

	Matrix weights1;
	Matrix* weightGradient1;
	Matrix weights2;
	Matrix* weightGradient2;

	Matrix* A1;
	Matrix* A1Grad;
	Matrix* A2;
	Matrix* A2Grad;
	Matrix* Ao;
	Matrix* AoGrad;

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

	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);
	int getNumParameters();
};

