#pragma once
#include "Layer2D.h"

class Dense2D : public Layer2D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer;

	Matrix weights;
	Matrix* weightGradient;
	Matrix* linearCombo;
	Matrix* backPropIntermediate;

	Activation* activation;
	Optimizer* optimizer;

	Dense2D(Activation* activation, int size);

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

