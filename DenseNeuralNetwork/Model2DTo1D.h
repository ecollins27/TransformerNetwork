#pragma once
#include "Input2D.h"
#include "Loss.h"
#include <climits>
#include <thread>
#include <functional>

class Model2DTo1D {

public:

	static int NUM_CORES;

	Input2D* inputLayer;
	Layer* tempLayer;
	Layer1D* outputLayer;
	int t;

	Model2DTo1D(int inputSize);

	void addLayer(Layer* layer);

	template<class T>
	T* getLayer(int index) {
		Layer* layer = inputLayer;
		for (int i = 0; i < index; i++) {
			layer = layer->nextLayer;
		}
		return (T*)layer;
	}

	int getNumParameters();
	void addTransformerBlock(int numHeads, int keySize, int valueSize);

	void fit(Loss* lossFunction, int numData, int* numTokens, float*** X, float** y, int numMetrics, Loss** metrics, TrainingParams* params);
	void test(Loss* lossFunction, int numData, int* numTokens, float*** X, float** y, int numMetrics, Loss** metrics);
	void save(string fileName);

private:
	void applyGradients(float learningRate);
	void updateAverages(Loss* lossFunction, float** y, float* averages, int numMetrics, Loss** metrics);
	void predict(float** input, int thread);
	void evaluateValidation(Loss* lossFunction, int valSize, float*** XVal, float** yVal, int* numTokens, int batchSize, int numMetrics, Loss** metrics);
	void forwardPropagate(float** input, int thread);
	void backPropagate(Loss* lossFunction, float** yTrue, int thread);
	void shuffle(int numData, int* numTokens, float*** X, float** y);
};

