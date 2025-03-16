#pragma once
#include "Layer2D.h"

class TransformerAttention : public Layer2D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer;

	int numHeads, keySize, valueSize;

	Matrix* Wq;
	Matrix** WqGrad;
	Matrix* Wk;
	Matrix** WkGrad;
	Matrix* Wv;
	Matrix** WvGrad;
	Matrix Wo;
	Matrix* WoGrad;

	Matrix** Q;
	Matrix** QGrad;
	Matrix** K;
	Matrix** KGrad;
	Matrix** V;
	Matrix** VGrad;

	Matrix** A;
	Matrix** AGrad;

	Matrix* Ac;
	Matrix** AcSub;
	Matrix* AcGrad;
	Matrix** AcSubGrad;

	Activation* softmax;

	Optimizer* outputOptimizer;
	Optimizer** keyOptimizers;
	Optimizer** queryOptimizers;
	Optimizer** valueOptimizers;

	TransformerAttention(int numHeads, int keySize, int valueSize);

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

