#pragma once
#include "Layer2D.h"

class TransformerAttention : public Layer2D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer = NULL;

	int numHeads, keySize, valueSize;

	MatrixBatch* prevNeuronGradient;

	MatrixBatch Wq;
	MatrixBatch WqGrad;
	MatrixBatch Wk;
	MatrixBatch WkGrad;
	MatrixBatch Wv;
	MatrixBatch WvGrad;
	Matrix Wo;
	Matrix WoGrad;

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

	Matrix* Ac;
	MatrixBatch* AcSub;
	Matrix* AcGrad;
	MatrixBatch* AcSubGrad;

	Activation* softmax;

	Optimizer<Matrix>* outputOptimizer;
	Optimizer<MatrixBatch>* keyOptimizers;
	Optimizer<MatrixBatch>* queryOptimizers;
	Optimizer<MatrixBatch>* valueOptimizers;

	TransformerAttention(int numHeads, int keySize, int valueSize);

	void initPropagationQueue(OperationQueue& queue);
	void initBackPropQueue(OperationQueue& queue);
	void initApplicationQueue(OperationQueue& queue, float learningRate, int& t);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);

	void setNumTokens(int* numTokens);
	void setOptimizer(Optimizer<>* optimizer);
	int getNumParameters();
};

