#pragma once
#include "TrainingParams.h"
#include "Matrix.h"

class Optimizer {

public:
	static Optimizer* GRADIENT_DESCENT;
	static Optimizer* MOMENTUM;
	static Optimizer* ADAM;
	static Optimizer* ADEMAMIX;

	int height, width;

	virtual void applyGradient(double** weightGradient, double** weights, double t, TrainingParams* params) = 0;
	virtual Optimizer* clone() = 0;
	virtual void setDimensions(int height, int width) = 0;
};

class GradientDescent : public Optimizer {

public:
	void applyGradient(double** weightGradient, double** weights, double t, TrainingParams* params);
	Optimizer* clone();
	void setDimensions(int height, int width);

};

class Momentum : public Optimizer {

public:
	double beta;
	double** M;

	Momentum(double beta);
	void applyGradient(double** weightGradient, double** weights, double t, TrainingParams* params);
	Optimizer* clone();
	void setDimensions(int height, int width);

};

class Adam : public Optimizer {

public:
	double beta1, beta2, lambda;
	double** M;
	double** S;

	Adam(double beta1, double beta2, double lambda);
	void applyGradient(double** weightGradient, double** weights, double t, TrainingParams* params);
	Optimizer* clone();
	void setDimensions(int height, int width);

};

class AdEMAMix : public Optimizer {

public:

	double beta1, beta2, beta3, alpha, lambda;
	double** M1;
	double** M2;
	double** S;

	AdEMAMix(double beta1, double beta2, double beta3, double alpha, double lambda);
	void applyGradient(double** weightGradient, double** weights, double t, TrainingParams* params);
	Optimizer* clone();
	void setDimensions(int height, int width);
};

