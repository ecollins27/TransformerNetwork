#pragma once
#include "Input2D.h"
#include "Loss1D.h"
#include <climits>
#include <functional>
#include "Dataset.h"
#include "Model.h"
#include <chrono>

using namespace std::chrono;

class Model2DTo1D : public Model {

public:
	const static string MODEL_NAME;
	static int NUM_CORES;

	Input2D* inputLayer = NULL;
	Layer* tempLayer = NULL;
	Layer1D* outputLayer = NULL;
	int t;
	atomic<int> forwardThreadCount;
	atomic<int> progress;

	const int MAX_NUM_TOKENS = 250;

	Model2DTo1D(int inputSize);

	void addLayer(Layer* layer);

	Layer* getLayer(int index);

	int getNumParameters();
	void addTransformer(int numHeads, int keySize, int valueSize);

	void fit(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics, TrainingParams* params);
	void test(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics);
	void save(string fileName);
	void printLayers();

private:
	void formatData(Dataset*& trainingData, Dataset*& valData, int& trainingNum, int& valNum, float valSplit, bool useSplitVal, int batchSize);
	string estimateTime(auto start, float progress);
	void formatData(Dataset*& data, Dataset*& trainingData, Dataset*& valData, int& trainingNum, int& valNum, int& maxTokenSize, int& valMaxTokenSize, float valSplit, bool useSplitVal, int batchSize);
	void updateAverages(Loss1D* lossFunction, float** y, atomic<float>* averages, int numMetrics, Loss1D** metrics);
	void evaluateValidation(OperationQueue* predict, Loss1D* lossFunction, int valNum, Dataset* valData, int batchSize, int numMetrics, Loss1D** metrics, atomic<float>* averages, int threadID, barrier<>* sync);
	void threadFit(int trainingNum, Dataset* trainingData, int valNum, Dataset* valData, Loss1D* lossFunction, OperationQueue* forwardProp, OperationQueue* backProp, OperationQueue* applyGradients, OperationQueue* predict, int numMetrics, Loss1D** metrics, string header, atomic<float>* averages, float learningRate, int batchSize, int threadID, barrier<>* sync);

	Dataset* partitionData(Dataset* data);
};

