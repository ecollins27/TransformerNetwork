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
	void predict(double** input);
	void forwardPropagate(double** input);
	void backPropagate(Loss* lossFunction, double** yTrue);
	void applyGradients(double learningRate);
	void fit(Loss* lossFunction, double** X, double** y, double* losses, int numMetrics, Loss** metrics, TrainingParams* params);
	void fit(Loss* lossFunction, int numData, double** X, double** y, int numMetrics, Loss** metrics, TrainingParams* params);
	void test(Loss* lossFunction, int numData, double** X, double** y, int numMetrics, Loss** metrics);
	void shuffle(int numData, double** X, double** y);
	void setTrainable(bool trainable);
	void save(string fileName);
	int getNumParameters();
};

