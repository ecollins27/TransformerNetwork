#include "Optimizer.h"

Optimizer* Optimizer::GRADIENT_DESCENT = { new GradientDescent(0) };
Optimizer* Optimizer::MOMENTUM = { new Momentum(0.9, 0) };
Optimizer* Optimizer::ADAM = { new Adam(0.9,0.999, 0) };
Optimizer* Optimizer::ADEMAMIX = { new AdEMAMix(0.9, 0.9999, 0.999, 5, 0) };

GradientDescent::GradientDescent(float regConstant) {
	this->regConstant = regConstant;
}

void GradientDescent::applyGradient(float** weightGradient, float** weights, float t, float learningRate) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			weights[i][j] -= learningRate * (weightGradient[i][j] + 2 * regConstant * weights[i][j]);
			weightGradient[i][j] = 0;
		}
	}
}

Optimizer* GradientDescent::clone() {
	return { new GradientDescent(regConstant) };
}

void GradientDescent::setDimensions(int height, int width) {
	this->height = height;
	this->width = width;
	return;
}

Momentum::Momentum(float beta, float regConstant) {
	this->beta = beta;
	this->regConstant = regConstant;
}

void Momentum::applyGradient(float** weightGradient, float** weights, float t, float learningRate) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			M[i][j] = beta * M[i][j] - learningRate * (weightGradient[i][j] + 2 * regConstant * weights[i][j]);
			weights[i][j] += M[i][j];
			weightGradient[i][j] = 0;
		}
	}
}

Optimizer* Momentum::clone() {
	return { new Momentum(beta, regConstant) };
}

void Momentum::setDimensions(int height, int width) {
	this->height = height;
	this->width = width;
	M = Matrix::allocateMatrix(Matrix::ZERO_FILL, height, width);
}

Adam::Adam(float beta1, float beta2, float regConstant) {
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->regConstant = regConstant;
}

void Adam::applyGradient(float** weightGradient, float** weights, float t, float learningRate) {
	float mScalar = 1.0 / (1 - pow(beta1, t));
	float sScalar = 1.0 / (1 - pow(beta2, t));
	float fullGradient;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			fullGradient = weightGradient[i][j] + 2 * regConstant * weights[i][j];
			M[i][j] = beta1 * M[i][j] + (1 - beta1) * fullGradient;
			S[i][j] = beta2 * S[i][j] + (1 - beta2) * fullGradient * fullGradient;
			weights[i][j] -= learningRate * mScalar * M[i][j] / sqrt(sScalar * S[i][j] + 0.0000001);
			weightGradient[i][j] = 0;
		}
	}
}

Optimizer* Adam::clone() {
	return { new Adam(beta1, beta2, regConstant) };
}

void Adam::setDimensions(int height, int width) {
	this->height = height;
	this->width = width;
	M = Matrix::allocateMatrix(Matrix::ZERO_FILL, height, width);
	S = Matrix::allocateMatrix(Matrix::ZERO_FILL, height, width);
}

AdEMAMix::AdEMAMix(float beta1, float beta2, float beta3, float alpha, float regConstant) {
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->beta3 = beta3;
	this->alpha = alpha;
	this->regConstant = regConstant;
}

void AdEMAMix::applyGradient(float** weightGradient, float** weights, float t, float learningRate) {
	float mScalar = 1.0 / (1 - pow(beta1, t));
	float sScalar = 1.0 / (1 - pow(beta2, t));
	float fullGradient;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			fullGradient = weightGradient[i][j] + 2 * regConstant * weights[i][j];
			M1[i][j] = beta1 * M1[i][j] + (1 - beta1) * fullGradient;
			M2[i][j] = beta3 * M2[i][j] + (1 - beta3) * fullGradient;
			S[i][j] = beta2 * S[i][j] + (1 - beta2) * fullGradient * fullGradient;
			weights[i][j] -= learningRate * (mScalar * M1[i][j] + alpha * M2[i][j]) / sqrt(sScalar * S[i][j] + 0.0000001);
			weightGradient[i][j] = 0;
		}
	}
}

Optimizer* AdEMAMix::clone(){
	return { new AdEMAMix(beta1, beta2, beta3, alpha, regConstant) };
}

void AdEMAMix::setDimensions(int height, int width) {
	this->height = height;
	this->width = width;
	M1 = Matrix::allocateMatrix(Matrix::ZERO_FILL, height, width);
	M2 = Matrix::allocateMatrix(Matrix::ZERO_FILL, height, width);
	S = Matrix::allocateMatrix(Matrix::ZERO_FILL, height, width);
}