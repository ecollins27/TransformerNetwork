#pragma once
#include "Input1D.h"
#include "Loss1D.h"
#include "Model.h"
#include <barrier>

class Model1D : public Model {

public:
	const static string MODEL_NAME;

	Input1D* inputLayer = NULL;
	Layer1D* outputLayer = NULL;
	int t;

	Model1D(int inputSize);

	void addLayer(Layer* layer);

	Layer* getLayer(int index);

	int getNumParameters();
	void fit(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics, TrainingParams* params);
	void test(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics);
	void save(string filename);

	void printLayers();

private:
	void formatData(Dataset*& trainingData, Dataset*& valData, int& trainingNum, int& valNum, float valSplit, bool useSplitVal, int batchSize);
	void threadFit(int trainingNum, Dataset* trainingData, int valNum, Dataset* valData, Loss1D* lossFunction, OperationQueue* forwardProp, OperationQueue* backProp, OperationQueue* applyGradients, OperationQueue* predict, int numMetrics, Loss1D** metrics, string header, atomic<float>* averages, float learningRate, int batchSize, int threadID, barrier<>* sync);
	void updateAverages(Loss1D* lossFunction, float** y, atomic<float>* averages, int numMetrics, Loss1D** metrics);
	void evaluateValidation(OperationQueue* predict, Loss1D* lossFunction, int valNum, Dataset* valData, int batchSize, int numMetrics, Loss1D** metrics, atomic<float>* averages, int threadID, barrier<>* sync);
};

