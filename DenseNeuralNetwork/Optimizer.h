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
	float weightDecay;

	virtual void applyGradient(float** weightGradient, float** weights, float t, float learningRate) = 0;
	virtual Optimizer* clone() = 0;
	virtual void setDimensions(int height, int width) = 0;
};

class GradientDescent : public Optimizer {

public:
	GradientDescent(float weightDecay);
	void applyGradient(float** weightGradient, float** weights, float t, float learningRate);
	Optimizer* clone();
	void setDimensions(int height, int width);

};

class Momentum : public Optimizer {

public:
	float beta;
	float** M;

	Momentum(float beta, float weightDecay);
	void applyGradient(float** weightGradient, float** weights, float t, float learningRate);
	Optimizer* clone();
	void setDimensions(int height, int width);

};

class Adam : public Optimizer {

public:
	float beta1, beta2;
	float** M;
	float** S;

	Adam(float beta1, float beta2, float weightDecay);
	void applyGradient(float** weightGradient, float** weights, float t, float learningRate);
	Optimizer* clone();
	void setDimensions(int height, int width);

};

class AdEMAMix : public Optimizer {

public:

	float beta1, beta2, beta3, alpha;
	float** M1;
	float** M2;
	float** S;

	AdEMAMix(float beta1, float beta2, float beta3, float alpha, float weightDecay);
	void applyGradient(float** weightGradient, float** weights, float t, float learningRate);
	Optimizer* clone();
	void setDimensions(int height, int width);
};

