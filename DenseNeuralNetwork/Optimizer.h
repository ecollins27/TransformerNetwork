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
	Matrix weightGradient;
	float regConstant;

	virtual void applyGradient(Matrix weights, float t, float learningRate, int batchSize) = 0;
	virtual Optimizer* clone() = 0;
	virtual void setDimensions(int height, int width) = 0;
	void addGradient(Matrix gradient);

};

class GradientDescent : public Optimizer {

public:
	GradientDescent(float weightDecay);
	void applyGradient(Matrix weights, float t, float learningRate, int batchSize);
	Optimizer* clone();
	void setDimensions(int height, int width);

};

class Momentum : public Optimizer {

public:
	float beta;
	Matrix M;

	Momentum(float beta, float weightDecay);
	void applyGradient(Matrix weights, float t, float learningRate, int batchSize);
	Optimizer* clone();
	void setDimensions(int height, int width);

};

class Adam : public Optimizer {

public:
	float beta1, beta2;
	Matrix M;
	Matrix S;

	Adam(float beta1, float beta2, float weightDecay);
	void applyGradient(Matrix weights, float t, float learningRate, int batchSize);
	Optimizer* clone();
	void setDimensions(int height, int width);

};

class AdEMAMix : public Optimizer {

public:

	float beta1, beta2, beta3, alpha;
	Matrix M1;
	Matrix M2;
	Matrix S;

	AdEMAMix(float beta1, float beta2, float beta3, float alpha, float weightDecay);
	void applyGradient(Matrix weights, float t, float learningRate, int batchSize);
	Optimizer* clone();
	void setDimensions(int height, int width);
};

