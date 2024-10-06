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
	double weightDecay;

	virtual void applyGradient(double** weightGradient, double** weights, double t, double learningRate) = 0;
	virtual Optimizer* clone() = 0;
	virtual void setDimensions(int height, int width) = 0;
};

class GradientDescent : public Optimizer {

public:
	GradientDescent(double weightDecay);
	void applyGradient(double** weightGradient, double** weights, double t, double learningRate);
	Optimizer* clone();
	void setDimensions(int height, int width);

};

class Momentum : public Optimizer {

public:
	double beta;
	double** M;

	Momentum(double beta, double weightDecay);
	void applyGradient(double** weightGradient, double** weights, double t, double learningRate);
	Optimizer* clone();
	void setDimensions(int height, int width);

};

class Adam : public Optimizer {

public:
	double beta1, beta2;
	double** M;
	double** S;

	Adam(double beta1, double beta2, double weightDecay);
	void applyGradient(double** weightGradient, double** weights, double t, double learningRate);
	Optimizer* clone();
	void setDimensions(int height, int width);

};

class AdEMAMix : public Optimizer {

public:

	double beta1, beta2, beta3, alpha;
	double** M1;
	double** M2;
	double** S;

	AdEMAMix(double beta1, double beta2, double beta3, double alpha, double weightDecay);
	void applyGradient(double** weightGradient, double** weights, double t, double learningRate);
	Optimizer* clone();
	void setDimensions(int height, int width);
};

