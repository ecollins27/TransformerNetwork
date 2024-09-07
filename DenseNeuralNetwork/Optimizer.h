#pragma once
#include "TrainingParams.h"
#include "Matrix.h"

class Optimizer {

public:
	static Optimizer* GRADIENT_DESCENT;
	static Optimizer* MOMENTUM;
	static Optimizer* ADAM;

	virtual void applyGradient(int m, int n, double** weightM, double** weightS, double** weightGradient, double** weights, double t, TrainingParams* params) = 0;
};

class GradientDescent : public Optimizer {

public:
	void applyGradient(int m, int n, double** weightM, double** weightS, double** weightGradient, double** weights, double t, TrainingParams* params);

};

class Momentum : public Optimizer {

public:
	double beta;

	Momentum(double beta);
	void applyGradient(int m, int n, double** weightM, double** weightS, double** weightGradient, double** weights, double t, TrainingParams* params);

};

class Adam : public Optimizer {

public:
	double beta1, beta2;

	Adam(double beta1, double beta2);
	void applyGradient(int m, int n, double** weightM, double** weightS, double** weightGradient, double** weights, double t, TrainingParams* params);

};

