#pragma once
#include "Layer.h"
#include "Activation.h"

class MultiHeadAttentionLayer : public Layer {

public:

	int numHeads, keySize, valueSize;

	float*** Wq;
	float*** WqGrad;
	float*** Wk;
	float*** WkGrad;
	float*** Wv;
	float*** WvGrad;
	float** Wo;
	float** WoGrad;

	float*** Q;
	float*** QGrad;
	float*** K;
	float*** KGrad;
	float*** V;
	float*** VGrad;
	float*** A;
	float*** AGrad;
	float*** Ao;
	float*** AoGrad;
	float** Ac;
	float** AcTrans;
	float** AcGrad;
	float** AcGradTrans;
	float**** activationGradients;

	Optimizer** queryOptimizers;
	Optimizer** keyOptimizers;
	Optimizer** valueOptimizers;

	Activation* softmax = Activation::SOFTMAX->clone();

	MultiHeadAttentionLayer(int numHeads, int keySize, int valueSize);
	void predict();
	void forwardPropagate();
	void backPropagate();

	void setPrevLayer(Layer* prevLayer);
	void setNextLayer(Layer* nextLayer);
	void setBatchSize(int batchSize);
	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);

	void save(ofstream& file);
	int getNumParameters();
	void setMaxBatchSize(int maxBatchSize);
};

