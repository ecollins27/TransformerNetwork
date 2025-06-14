#pragma once
#include "Layer1D.h"

class Loss1D {

public:

	virtual float loss(Layer1D* layer, float** yTrue) = 0;
	virtual void differentiate(Layer1D* layer, float** yTrue) = 0;
	virtual string toString() = 0;

};

class MeanSquaredError1D : public Loss1D {

public:
	float loss(Layer1D* layer, float** yTrue);
	void differentiate(Layer1D* layer, float** yTrue);
	string toString();
};

class BinaryCrossEntropy1D : public Loss1D {

public:
	float loss(Layer1D* layer, float** yTrue);
	void differentiate(Layer1D* layer, float** yTrue);
	string toString();
};

class CategoricalCrossEntropy1D : public Loss1D {

public:
	float loss(Layer1D* layer, float** yTrue);
	void differentiate(Layer1D* layer, float** yTrue);
	string toString();
};

class Accuracy1D : public Loss1D {

public:
	float loss(Layer1D* layer, float** yTrue);
	void differentiate(Layer1D* layer, float** yTrue);
	string toString();
};

class BinaryAccuracy1D : public Loss1D {

public:
	float loss(Layer1D* layer, float** yTrue);
	void differentiate(Layer1D* layer, float** yTrue);
	string toString();
};
