#pragma once
#include "Input2D.h"
#include "Loss1D.h"
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

	void fit(Loss1D* lossFunction, int numData, int* numTokens, float*** X, float** y, int numMetrics, Loss1D** metrics, TrainingParams* params);
	void test(Loss1D* lossFunction, int numData, int* numTokens, float*** X, float** y, int numMetrics, Loss1D** metrics);
	void save(string fileName);

private:
	void applyGradients(float learningRate);
	void updateAverages(Loss1D* lossFunction, float** y, float* averages, int numMetrics, Loss1D** metrics);
	void predict(float** input, int thread);
	void evaluateValidation(Loss1D* lossFunction, int valSize, float*** XVal, float** yVal, int* numTokens, int batchSize, int numMetrics, Loss1D** metrics);
	void forwardPropagate(float** input, int thread);
	void backPropagate(Loss1D* lossFunction, float** yTrue, int thread);
	void shuffle(int numData, int* numTokens, float*** X, float** y);
};

