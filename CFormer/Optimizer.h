#pragma once
#include "TrainingParams.h"
#include "Matrix2.h"
#include "OptimizerBatch.h"

class Optimizer {

public:
	static Optimizer* GRADIENT_DESCENT;
	static Optimizer* MOMENTUM;
	static Optimizer* ADAM;
	static Optimizer* ADEMAMIX;

	int height, width, batchSize;
	Matrix2 weightGradient;
	float regConstant;
	MatrixBatch weightGradients;

	virtual void applyGradient(Matrix2& weights, float t, float learningRate) = 0;
	virtual Optimizer* clone() = 0;
	virtual OptimizerBatch* cloneBatch() = 0;
	virtual void setDimensions(int height, int width) = 0;
	void setBatchSize(int batchSize, Matrix2* gradients);
	void condenseGradients();
};

class GradientDescent : public Optimizer {

public:
	GradientDescent(float weightDecay);
	void applyGradient(Matrix2& weights, float t, float learningRate);
	Optimizer* clone();
	OptimizerBatch* cloneBatch();
	void setDimensions(int height, int width);

};

class Momentum : public Optimizer {

public:
	float beta;
	Matrix2 M;

	Momentum(float beta, float weightDecay);
	void applyGradient(Matrix2& weights, float t, float learningRate);
	Optimizer* clone();
	OptimizerBatch* cloneBatch();
	void setDimensions(int height, int width);

};

class Adam : public Optimizer {

public:
	float beta1, beta2;
	Matrix2 M;
	Matrix2 S;

	Adam(float beta1, float beta2, float weightDecay);
	void applyGradient(Matrix2& weights, float t, float learningRate);
	Optimizer* clone();
	OptimizerBatch* cloneBatch();
	void setDimensions(int height, int width);

};

class AdEMAMix : public Optimizer {

public:

	float beta1, beta2, beta3, alpha;
	Matrix2 M1;
	Matrix2 M2;
	Matrix2 S;

	AdEMAMix(float beta1, float beta2, float beta3, float alpha, float weightDecay);
	void applyGradient(Matrix2& weights, float t, float learningRate);
	Optimizer* clone();
	OptimizerBatch* cloneBatch();
	void setDimensions(int height, int width);
};