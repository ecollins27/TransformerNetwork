#pragma once
#include "Loss.h"
#include "DenseLayer.h"
#include "Dropout.h"
#include <fstream>

class NeuralNetwork {
public:
	InputLayer* inputLayer;
	Layer* outputLayer;
	int t;

	NeuralNetwork(int inputSize);
	NeuralNetwork(string fileName);

	void addLayer(Layer* layer);
	void predict(float** input);
	void forwardPropagate(float** input);
	void backPropagate(Loss* lossFunction, float** yTrue);
	void applyGradients(float learningRate);
	void fit(Loss* lossFunction, float** X, float** y, float* losses, int numMetrics, Loss** metrics, TrainingParams* params);
	void fit(Loss* lossFunction, int numData, float** X, float** y, int numMetrics, Loss** metrics, TrainingParams* params);
	void test(Loss* lossFunction, int numData, float** X, float** y, int numMetrics, Loss** metrics);
	void shuffle(int numData, float** X, float** y);
	void setTrainable(bool trainable);
	void save(string fileName);
	int getNumParameters();
};

