#pragma once
#include "Layer.h"
#include "Activation.h"

class MultiAttentionLayer : public Layer {

public:

	int numHeads, keySize, valueSize, maxTokenSize;

	//H x Dk x D
	float*** queryWeights;
	float*** queryWeightGradients;
	//H x Dk x D
	float*** keyWeights;
	float*** keyWeightGradients;
	//H x Dv x D
	float*** valueWeights;
	float*** valueWeightGradients;
	//HDv x D
	float** outputWeights;
	float** outputWeightGradient;

	//N x Dk
	float*** queries;
	//N x Dk
	float*** keys;
	//Dv x N
	float*** values;
	//H x N x N
	float*** activations;
	//H x N x N
	float*** activationOutputs;
	//N x HDv
	float** attentionConcat;
	//HDv x N
	float** attentionConcatTranspose;
	//H x N x N x N
	float**** activationGradients;

	Optimizer** queryOptimizers;
	Optimizer** keyOptimizers;
	Optimizer** valueOptimizers;

	Activation* softmax = Activation::SOFTMAX->clone();

	MultiAttentionLayer(int numHeads, int keySize, int valueSize, int maxTokenSize);
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
};

