#pragma once
#include "Loss.h"
#include "DenseLayer.h"
#include <fstream>

class NeuralNetwork {
public:
	InputLayer* inputLayer;
	Layer* outputLayer;
	Loss* lossFunction;
	int t;

	NeuralNetwork(Loss* lossFunction, int inputSize);
	NeuralNetwork(string fileName);

	void addLayer(Layer* layer);
	void forwardPropagate(double* input);
	void backPropagate(double* yTrue);
	void applyGradients(TrainingParams* params);
	double getLoss(double* yTrue);
	void fit(double* X, double* y, double* losses, TrainingParams* params);
	void fit(int numData, double** X, double** y, TrainingParams* params);
	double test(int numData, double** X, double** y, int numMetrics, Loss** metrics);
	void shuffle(int numData, double** X, double** y);
	void setTrainable(bool trainable);
	void save(string fileName);
};

