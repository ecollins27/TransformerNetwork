#pragma once
#include "Input1D.h"
#include "Loss1D.h"

class Model1D {

public:
	Input1D* inputLayer;
	Layer1D* outputLayer;
	int t;

	Model1D(int inputSize);

	void addLayer(Layer1D* layer);
	template<typename T>
	T* getLayer(int index) {
		Layer* layer = inputLayer;
		for (int i = 0; i < index; i++) {
			layer = layer->nextLayer;
		}
		return (T*)layer;
	}

	int getNumParameters();
	void fit(Loss1D* lossFunction, int numData, float** X, float** y, int numMetrics, Loss1D** metrics, TrainingParams* params);
	void test(Loss1D* lossFunction, int numData, float** X, float** y, int numMetrics, Loss1D** metrics);
	void save(string filename);

private:
	void updateAverages(Loss1D* lossFunction, float** y, float* averages, int numMetrics, Loss1D** metrics);
	void evaluateValidation(Loss1D* lossFunction, int numData, float** X, float** y, int batchSize, int numMetrics, Loss1D** metrics);
	void applyGradients(float learningRate);
	void predict(float** input);
	void forwardPropagate(float** input);
	void backPropagate(Loss1D* lossFunction, float** yTrue);
	void shuffle(int numData, float** X, float** y);
};

