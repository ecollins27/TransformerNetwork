#pragma once
#include "Layer2D.h"

class MultiHeadAttention : public Layer2D {

public:
	Layer2D* prevLayer;

	int numHeads, keySize, valueSize;

	Matrix2* Wq;
	Matrix2** WqGrad;
	Matrix2* Wk;
	Matrix2** WkGrad;
	Matrix2* Wv;
	Matrix2** WvGrad;
	Matrix2 Wo;
	Matrix2* WoGrad;

	Matrix2** Q;
	Matrix2** QGrad;
	Matrix2** K;
	Matrix2** KGrad;
	Matrix2** V;
	Matrix2** VGrad;

	Matrix2** A;
	Matrix2** AGrad;

	Matrix2** Ao;
	Matrix2** AoGrad;

	Matrix2* Ac;
	Matrix2** AcSub;
	Matrix2* AcGrad;
	Matrix2* AcSubGrad;

	Matrix3D** activationGradients;

	Activation* softmax;

	Optimizer* outputOptimizer;
	Optimizer** keyOptimizers;
	Optimizer** queryOptimizers;
	Optimizer** valueOptimizers;

	MultiHeadAttention(int numHeads, int keySize, int valueSize);

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);

	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);
	int getNumParameters();
};

