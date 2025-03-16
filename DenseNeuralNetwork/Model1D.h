#pragma once
#include "Input1D.h"
#include "Loss1D.h"
#include "Model.h"

class Model1D : public Model {

public:
	const static string MODEL_NAME;

	Input1D* inputLayer;
	Layer1D* outputLayer;
	int t;

	Model1D(int inputSize);

	void addLayer(Layer* layer);

	Layer* getLayer(int index);

	int getNumParameters();
	void fit(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics, TrainingParams* params);
	void test(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics);
	void save(string filename);

private:
	void updateAverages(Loss1D* lossFunction, float** y, float* averages, int numMetrics, Loss1D** metrics);
	void evaluateValidation(Loss1D* lossFunction, Dataset* valData, int batchSize, int numMetrics, Loss1D** metrics);
	void applyGradients(float learningRate);
	void predict(void* input, bool sparse);
	void forwardPropagate(void* input, bool sparse);
	void backPropagate(Loss1D* lossFunction, float** yTrue);
};

