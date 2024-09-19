#include "Optimizer.h"

Optimizer* Optimizer::GRADIENT_DESCENT = { new GradientDescent() };
Optimizer* Optimizer::MOMENTUM = { new Momentum(0.9) };
Optimizer* Optimizer::ADAM = { new Adam(0.9,0.999, 0) };
Optimizer* Optimizer::ADEMAMIX = { new AdEMAMix(0.9, 0.9999, 0.999, 5, 0) };

void GradientDescent::applyGradient(double** weightGradient, double** weights, double t, TrainingParams* params) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			weights[i][j] -= params->learningRate * weightGradient[i][j];
			weightGradient[i][j] = 0;
		}
	}
}

Optimizer* GradientDescent::clone() {
	return { new GradientDescent() };
}

void GradientDescent::setDimensions(int height, int width) {
	this->height = height;
	this->width = width;
	return;
}

Momentum::Momentum(double beta) {
	this->beta = beta;
}

void Momentum::applyGradient(double** weightGradient, double** weights, double t, TrainingParams* params) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			M[i][j] = beta * M[i][j] - params->learningRate * weightGradient[i][j];
			weights[i][j] += M[i][j];
			weightGradient[i][j] = 0;
		}
	}
}

Optimizer* Momentum::clone() {
	return { new Momentum(beta) };
}

void Momentum::setDimensions(int height, int width) {
	this->height = height;
	this->width = width;
	M = Matrix::allocateMatrix(Matrix::ZERO_FILL, height, width);
}

Adam::Adam(double beta1, double beta2, double lambda) {
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->lambda = lambda;
}

void Adam::applyGradient(double** weightGradient, double** weights, double t, TrainingParams* params) {
	double mScalar = 1.0 / (1 - pow(beta1, t));
	double sScalar = 1.0 / (1 - pow(beta2, t));
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			M[i][j] = beta1 * M[i][j] + (1 - beta1) * weightGradient[i][j];
			S[i][j] = beta2 * S[i][j] + (1 - beta2) * weightGradient[i][j] * weightGradient[i][j];
			weights[i][j] -= params->learningRate * (lambda * weights[i][j] + mScalar * M[i][j] / sqrt(sScalar * S[i][j] + 0.0000001));
			weightGradient[i][j] = 0;
		}
	}
}

Optimizer* Adam::clone() {
	return { new Adam(beta1, beta2, lambda) };
}

void Adam::setDimensions(int height, int width) {
	this->height = height;
	this->width = width;
	M = Matrix::allocateMatrix(Matrix::ZERO_FILL, height, width);
	S = Matrix::allocateMatrix(Matrix::ZERO_FILL, height, width);
}

AdEMAMix::AdEMAMix(double beta1, double beta2, double beta3, double alpha, double lambda) {
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->beta3 = beta3;
	this->alpha = alpha;
	this->lambda = lambda;
}

void AdEMAMix::applyGradient(double** weightGradient, double** weights, double t, TrainingParams* params) {
	double mScalar = 1.0 / (1 - pow(beta1, t));
	double sScalar = 1.0 / (1 - pow(beta2, t));
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			M1[i][j] = beta1 * M1[i][j] + (1 - beta1) * weightGradient[i][j];
			M2[i][j] = beta3 * M2[i][j] + (1 - beta3) * weightGradient[i][j];
			S[i][j] = beta2 * S[i][j] + (1 - beta2) * weightGradient[i][j] * weightGradient[i][j];
			weights[i][j] -= params->learningRate * (lambda * weights[i][j] + (mScalar * M1[i][j] + alpha * M2[i][j]) / sqrt(sScalar * S[i][j] + 0.0000001));
			weightGradient[i][j] = 0;
		}
	}
}

Optimizer* AdEMAMix::clone(){
	return { new AdEMAMix(beta1, beta2, beta3, alpha, lambda) };
}

void AdEMAMix::setDimensions(int height, int width) {
	this->height = height;
	this->width = width;
	M1 = Matrix::allocateMatrix(Matrix::ZERO_FILL, height, width);
	M2 = Matrix::allocateMatrix(Matrix::ZERO_FILL, height, width);
	S = Matrix::allocateMatrix(Matrix::ZERO_FILL, height, width);
}