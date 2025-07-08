#pragma once
#include "Layer2D.h"

class TransformerAttention : public Layer2D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer = NULL;

	int numHeads, keySize, valueSize;

	MatrixBatch* prevNeuronGradient;

	MatrixBatch Wq;
	MatrixBatch* WqGrad;
	MatrixBatch Wk;
	MatrixBatch* WkGrad;
	MatrixBatch Wv;
	MatrixBatch* WvGrad;
	Matrix2 Wo;
	Matrix2* WoGrad;

	MatrixBatch* Q;
	MatrixBatch* QGrad;
	MatrixBatch* K;
	MatrixBatch* KGrad;
	MatrixBatch* V;
	MatrixBatch* VGrad;

	MatrixBatch* A;
	MatrixBatch* AGrad;
	MatrixBatch* Ao;
	MatrixBatch* AoGrad;

	Matrix2* Ac;
	MatrixBatch* AcSub;
	Matrix2* AcGrad;
	MatrixBatch* AcSubGrad;

	Activation* softmax;

	Optimizer* outputOptimizer;
	OptimizerBatch* keyOptimizers;
	OptimizerBatch* queryOptimizers;
	OptimizerBatch* valueOptimizers;

	TransformerAttention(int numHeads, int keySize, int valueSize);

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

