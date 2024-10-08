#pragma once
#include "InputLayer.h"
class Loss {

public:
	const static int NUM_LOSSES = 3;
	static Loss* MEAN_SQUARED_ERROR;
	static Loss* BINARY_CROSS_ENTROPY;
	static Loss* CATEGORICAL_CROSS_ENTROPY;
	static Loss* ACCURACY;
	static Loss* BINARY_ACCURACY;
	static Loss* ALL_LOSSES[NUM_LOSSES];

	virtual float loss(Layer* layer, float** yTrue) = 0;
	virtual void differentiate(Layer* layer, float** yTrue) = 0;
	virtual string toString() = 0;

};

class MeanSquaredError : public Loss {

public:
	float loss(Layer* layer, float** yTrue);
	void differentiate(Layer* layer, float** yTrue);
	string toString();
};

class BinaryCrossEntropy : public Loss {

public:
	float loss(Layer* layer, float** yTrue);
	void differentiate(Layer* layer, float** yTrue);
	string toString();
};

class CategoricalCrossEntropy : public Loss {

public:
	float loss(Layer* layer, float** yTrue);
	void differentiate(Layer* layer, float** yTrue);
	string toString();
};

class Accuracy : public Loss {

public:
	float loss(Layer* layer, float** yTrue);
	void differentiate(Layer* layer, float** yTrue);
	string toString();
};

class BinaryAccuracy : public Loss {

public:
	float loss(Layer* layer, float** yTrue);
	void differentiate(Layer* layer, float** yTrue);
	string toString();
};
