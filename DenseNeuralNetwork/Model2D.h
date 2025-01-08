#pragma once
#include "Input2D.h"
#include "Loss.h"
#include <climits>
#include <thread>

class Model2D {

public:

	Input2D* inputLayer;
	Layer* outputLayer;
	int t;
	bool output1D;

	Model2D(int inputSize);

	void addLayer(Layer* layer);

	template<class T>
	T* getLayer(int index) {
		Layer* layer = inputLayer;
		for (int i = 0; i < index; i++) {
			layer = layer->nextLayer;
		}
		return (T*)layer;
	}
	void applyGradients(float learningRate);
	int getNumParameters();
	void predict(float** input, int thread);
	void forwardPropagate(float** input, int thread);
	void backPropagate(Loss* lossFunction, float** yTrue, int thread);
	void addTransformerBlock(int numHeads, int keySize, int valueSize);
	void fitPoint(Loss* lossFunction, float** X, float** y, int thread);
	static void threadFit(Model2D* model, Loss* lossFunction, int trainingNum, int* numTokens, float*** X, float*** y, float* losses, int thread, int* batchProgress, int numMetrics, Loss** metrics, int batchSize, float learningRate, int epoch, int numEpochs);
	void oneThreadFit(Loss* lossFunction, int numData, int* numTokens, float*** X, float*** y,int numMetrics, Loss** metrics, TrainingParams* params, string filename);
	void fit(Loss* lossFunction, int numData, int* numTokens, float*** X, float*** y, int numMetrics, Loss** metrics, TrainingParams* params, string filename);
	void test(Loss* lossFunction, int numData, int* numTokens, float*** X, float*** y, int numMetrics, Loss** metrics);
	void shuffle(int numData, int* numTokens, float*** X, float*** y);
	void save(string fileName);
};

