#pragma once
#include "Input1D.h"
#include "Loss.h"

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
	void applyGradients(float learningRate);
	int getNumParameters();
	void predict(float** input);
	void forwardPropagate(float** input);
	void backPropagate(Loss* lossFunction, float** yTrue);
	void fit(Loss* lossFunction, float** X, float** y, float* losses, int numMetrics, Loss** metrics, TrainingParams* params);
	void fit(Loss* lossFunction, int numData, float** X, float** y, int numMetrics, Loss** metrics, TrainingParams* params);
	void test(Loss* lossFunction, int numData, float** X, float** y, int numMetrics, Loss** metrics);
	void shuffle(int numData, float** X, float** y);
	void save(string filename);
};

