#pragma once
#include "InputLayer.h"
class Loss {

public:
	static Loss* MEAN_SQUARED_ERROR;
	static Loss* BINARY_CROSS_ENTROPY;
	static Loss* CATEGORICAL_CROSS_ENTROPY;
	static Loss* ACCURACY;

	virtual double loss(Layer* layer, double* yTrue) = 0;
	virtual void differentiate(Layer* layer, double* yTrue) = 0;
	virtual string toString() = 0;

};

class MeanSquaredError : public Loss {

public:
	double loss(Layer* layer, double* yTrue);
	void differentiate(Layer* layer, double* yTrue);
	string toString();
};

class BinaryCrossEntropy : public Loss {

public:
	double loss(Layer* layer, double* yTrue);
	void differentiate(Layer* layer, double* yTrue);
	string toString();
};

class CategoricalCrossEntropy : public Loss {

public:
	double loss(Layer* layer, double* yTrue);
	void differentiate(Layer* layer, double* yTrue);
	string toString();
};

class Accuracy : public Loss {

public:
	double loss(Layer* layer, double* yTrue);
	void differentiate(Layer* layer, double* yTrue);
	string toString();
};
