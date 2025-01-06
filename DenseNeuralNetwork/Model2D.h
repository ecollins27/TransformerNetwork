#pragma once
#include "Input2D.h"
#include "Loss.h"
#include <climits>

class Model2D {

public:

	Input2D* inputLayer;
	Layer* outputLayer;
	int t;
	bool input1D;

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
	void fit(Loss* lossFunction, float** X, float** y, float* losses, int thread, int numMetrics, Loss** metrics, TrainingParams* params);
	void oneThreadFit(Loss* lossFunction, int numData, int* numTokens, float*** X, float*** y,int numMetrics, Loss** metrics, TrainingParams* params, string filename);
	void test(Loss* lossFunction, int numData, int* numTokens, float*** X, float*** y, int numMetrics, Loss** metrics);
	void shuffle(int numData, int* numTokens, float*** X, float*** y);
	void save(string fileName);
};

