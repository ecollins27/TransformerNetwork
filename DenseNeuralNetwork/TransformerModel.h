#pragma once
#include "Model.h"

class TransformerModel : public Model {

public:

	TransformerModel(int inputSize);

	void predict(float** input);
	void forwardPropagate(float** input);
	void backPropagate(Loss* lossFunction, float* yTrue);
	void fit(Loss* lossFunction, float** X, float* y, float* losses, int numMetrics, Loss** metrics, TrainingParams* params);
	void fit(Loss* lossFunction, int numData, float*** X, float** y, int numMetrics, Loss** metrics, TrainingParams* params);
	void test(Loss* lossFunction, int numData, float*** X, float** y, int numMetrics, Loss** metrics);
	void shuffle(int numData, float*** X, float** y);
	void save(string fileName);
};

