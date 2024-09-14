#pragma once
#include "TrainingParams.h"
#include "Matrix.h"

class Optimizer {

public:
	static Optimizer* GRADIENT_DESCENT;
	static Optimizer* MOMENTUM;
	static Optimizer* ADAM;
	static Optimizer* ADEMAMIX;

	virtual void applyGradient(int m, int n, double** weightM1, double** weightM2, double** weightS, double** weightGradient, double** weights, double t, TrainingParams* params) = 0;
};

class GradientDescent : public Optimizer {

public:
	void applyGradient(int m, int n, double** weightM1, double** weightM2, double** weightS, double** weightGradient, double** weights, double t, TrainingParams* params);

};

class Momentum : public Optimizer {

public:
	double beta;

	Momentum(double beta);
	void applyGradient(int m, int n, double** weightM1, double** weightM2, double** weightS, double** weightGradient, double** weights, double t, TrainingParams* params);

};

class Adam : public Optimizer {

public:
	double beta1, beta2, lambda;

	Adam(double beta1, double beta2, double lambda);
	void applyGradient(int m, int n, double** weightM1, double** weightM2, double** weightS, double** weightGradient, double** weights, double t, TrainingParams* params);

};

class AdEMAMix : public Optimizer {

public:

	double beta1, beta2, beta3, alpha, lambda;

	AdEMAMix(double beta1, double beta2, double beta3, double alpha, double lambda);
	void applyGradient(int m, int n, double** weightM1, double** weightM2, double** weightS, double** weightGradient, double** weights, double t, TrainingParams* params);
};

