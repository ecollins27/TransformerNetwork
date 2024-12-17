#pragma once
#include "Model.h"
#include "MultiHeadAttentionLayer.h"
#include "ResidualSave.h"
#include "ResidualAdd.h"
#include "DenseLayer.h"
#include "LayerNormalization.h"
#include <climits>

class TransformerModel : public Model {

public:

	TransformerModel(int inputSize);

	void predict(float** input);
	void forwardPropagate(float** input);
	void backPropagate(Loss* lossFunction, float* yTrue);
	void addTransformerBlock(int numHeads, int keySize, int valueSize);
	void fit(Loss* lossFunction, int numTokens, float** X, float* y, float* losses, int numMetrics, Loss** metrics, TrainingParams* params);
	void fit(Loss* lossFunction, int numData, int* numTokens, float*** X, float** y, int numMetrics, Loss** metrics, TrainingParams* params, string filename);
	void test(Loss* lossFunction, int numData, int* numTokens, float*** X, float** y, int numMetrics, Loss** metrics);
	void shuffle(int numData, int* numTokens, float*** X, float** y);
	void save(string fileName);
};

