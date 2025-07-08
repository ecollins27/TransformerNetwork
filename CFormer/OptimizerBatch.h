#pragma once
#include "TrainingParams.h"
#include "MatrixBatch.h"

class OptimizerBatch {

public:
	static OptimizerBatch* GRADIENT_DESCENT;
	static OptimizerBatch* MOMENTUM;
	static OptimizerBatch* ADAM;
	static OptimizerBatch* ADEMAMIX;

	int height, width, depth, batchSize;
	MatrixBatch weightGradient;
	float regConstant;
	float*** device;
	float*** hostDevice;
	bool condenseGradient = false;

	virtual void applyGradient(MatrixBatch& weights, float t, float learningRate) = 0;
	virtual OptimizerBatch* clone() = 0;
	virtual void setDimensions(int height, int width, int depth) = 0;
	void setBatchSize(int batchSize, MatrixBatch* gradients);
	void condenseGradients();
};

class GradientDescentBatch : public OptimizerBatch {

public:
	GradientDescentBatch(float weightDecay);
	void applyGradient(MatrixBatch& weights, float t, float learningRate);
	OptimizerBatch* clone();
	void setDimensions(int height, int width, int depth);

};

class MomentumBatch : public OptimizerBatch {

public:
	float beta;
	MatrixBatch M;

	MomentumBatch(float beta, float weightDecay);
	void applyGradient(MatrixBatch& weights, float t, float learningRate);
	OptimizerBatch* clone();
	void setDimensions(int height, int width, int depth);

};

class AdamBatch : public OptimizerBatch {

public:
	float beta1, beta2;
	MatrixBatch M;
	MatrixBatch S;

	AdamBatch(float beta1, float beta2, float weightDecay);
	void applyGradient(MatrixBatch& weights, float t, float learningRate);
	OptimizerBatch* clone();
	void setDimensions(int height, int width, int depth);

};

class AdEMAMixBatch : public OptimizerBatch {

public:

	float beta1, beta2, beta3, alpha;
	MatrixBatch M1;
	MatrixBatch M2;
	MatrixBatch S;

	AdEMAMixBatch(float beta1, float beta2, float beta3, float alpha, float weightDecay);
	void applyGradient(MatrixBatch& weights, float t, float learningRate);
	OptimizerBatch* clone();
	void setDimensions(int height, int width, int depth);
};