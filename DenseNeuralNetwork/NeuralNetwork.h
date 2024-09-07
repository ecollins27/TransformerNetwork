#pragma once
#include "Loss.h"

class NeuralNetwork {
public:
	InputLayer* inputLayer;
	Layer* outputLayer;
	Loss* lossFunction;
	int t;

	NeuralNetwork(Loss* lossFunction, int inputSize);

	void addLayer(Layer* layer);
	void forwardPropagate(double* input);
	void backPropagate(double* yTrue);
	void applyGradients(TrainingParams* params);
	double getLoss(double* yTrue);
	void fit(double* X, double* y, double* losses, TrainingParams* params);
	void fit(int numData, double** X, double** y, TrainingParams* params);
	void shuffle(int numData, double** X, double** y);
};

