#pragma once
#include "Layer1D.h"
#include "Layer2D.h"

class Loss {

public:

	virtual float loss(Layer* layer, float** yTrue, int thread, bool layer1D) = 0;
	virtual void differentiate(Layer* layer, float** yTrue, int thread, bool layer1D) = 0;
	virtual string toString() = 0;

};

class MeanSquaredError1D : public Loss {

public:
	float loss(Layer* layer, float** yTrue, int thread, bool layer1D);
	void differentiate(Layer* layer, float** yTrue, int thread, bool layer1D);
	string toString();
};

class BinaryCrossEntropy1D : public Loss {

public:
	float loss(Layer* layer, float** yTrue, int thread, bool layer1D);
	void differentiate(Layer* layer, float** yTrue, int thread, bool layer1D);
	string toString();
};

class CategoricalCrossEntropy1D : public Loss {

public:
	float loss(Layer* layer, float** yTrue, int thread, bool layer1D);
	void differentiate(Layer* layer, float** yTrue, int thread, bool layer1D);
	string toString();
};

class Accuracy1D : public Loss {

public:
	float loss(Layer* layer, float** yTrue, int thread, bool layer1D);
	void differentiate(Layer* layer, float** yTrue, int thread, bool layer1D);
	string toString();
};

class BinaryAccuracy1D : public Loss {

public:
	float loss(Layer* layer, float** yTrue, int thread, bool layer1D);
	void differentiate(Layer* layer, float** yTrue, int thread, bool layer1D);
	string toString();
};
