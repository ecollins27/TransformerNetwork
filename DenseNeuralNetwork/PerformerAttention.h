#pragma once
#include "Layer2D.h"

class PerformerAttention : public Layer2D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer;

	int numHeads, keySize, valueSize, M;

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
	Matrix** QK;
	Matrix** QKGrad;
	Matrix** K;
	Matrix** KGrad;
	Matrix** KK;
	Matrix** KKGrad;
	Matrix** V;
	Matrix** VGrad;

	Matrix** A;
	Matrix** AGrad;

	Matrix* Ac;
	Matrix** AcSub;
	Matrix* AcGrad;
	Matrix** AcSubGrad;

	Activation* softmax;

	Matrix omega;

	Optimizer* outputOptimizer;
	Optimizer** keyOptimizers;
	Optimizer** queryOptimizers;
	Optimizer** valueOptimizers;

	PerformerAttention(int numHeads, int keySize, int valueSize, int M);

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);

	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);
	int getNumParameters();
	void phi(int height, int width, Matrix X, Matrix Y);
};

