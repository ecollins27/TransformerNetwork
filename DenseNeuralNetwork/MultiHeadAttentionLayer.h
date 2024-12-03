#pragma once
#include "Layer.h"
#include "Activation.h"

class MultiHeadAttentionLayer : public Layer {

public:

	int numHeads, keySize, valueSize;

	float** neuronGradientTrans;

	float*** Wq;
	float*** WqTrans;
	float*** WqGrad;
	float*** Wk;
	float*** WkTrans;
	float*** WkGrad;
	float*** Wv;
	float*** WvTrans;
	float*** WvGrad;
	float** Wo;
	float** WoTrans;
	float** WoGrad;

	float*** Q;
	float*** QTrans;
	float*** QGrad;
	float*** QGradTrans;
	float*** K;
	float*** KTrans;
	float*** KGrad;
	float*** KGradTrans;
	float*** V;
	float*** VTrans;
	float*** VGrad;
	float*** VGradTrans;
	float*** A;
	float*** AGrad;
	float*** AGradTrans;
	float*** Ao;
	float*** AoTrans;
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

	void propagateLayer();
	void backPropagate();

	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);

	void save(ofstream& file);
	int getNumParameters();
	void setMaxBatchSize(int maxBatchSize);
};

