#pragma once
#include "NeuralNetwork.h"
#include <fstream>

class GAN {

public:
	double zero[1] = { 0.0 };
	double one[1] = { 1.0 };
	NeuralNetwork* generator;
	NeuralNetwork* discriminator;

	GAN(NeuralNetwork* generator, NeuralNetwork* discriminator);

	void forwardPropagate();
	void backPropagate(double* yTrue);
	void fit(double* X, double* losses, TrainingParams* params);
	void fit(int numData, double** X, TrainingParams* params);
	void setCoding();
	void writeToFile();
	void shuffle(int numData, double** X);
};

