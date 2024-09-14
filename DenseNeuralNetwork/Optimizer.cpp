#include "Optimizer.h"

Optimizer* Optimizer::GRADIENT_DESCENT = { new GradientDescent() };
Optimizer* Optimizer::MOMENTUM = { new Momentum(0.9) };
Optimizer* Optimizer::ADAM = { new Adam(0.9,0.999, 0) };
Optimizer* Optimizer::ADEMAMIX = { new AdEMAMix(0.9, 0.9999, 0.999, 5, 0) };

void GradientDescent::applyGradient(int m, int n, double** weightM1, double** weightM2, double** weightS, double** weightGradient, double** weights, double t, TrainingParams* params) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			weights[i][j] -= params->learningRate * weightGradient[i][j];
			weightGradient[i][j] = 0;
		}
	}
}

Momentum::Momentum(double beta) {
	this->beta = beta;
}

void Momentum::applyGradient(int m, int n, double** weightM1, double** weightM2, double** weightS, double** weightGradient, double** weights, double t, TrainingParams* params) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			weightM1[i][j] = beta * weightM1[i][j] - params->learningRate * weightGradient[i][j];
			weights[i][j] += weightM1[i][j];
			weightGradient[i][j] = 0;
		}
	}
}

Adam::Adam(double beta1, double beta2, double lambda) {
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->lambda = lambda;
}

void Adam::applyGradient(int m, int n, double** weightM1, double** weightM2, double** weightS, double** weightGradient, double** weights, double t, TrainingParams* params) {
	Matrix::add(m, n, weightM1, weightGradient, weightM1, beta1, (1 - beta1));
	double mScalar = 1.0 / (1 - pow(beta1, t));
	double sScalar = 1.0 / (1 - pow(beta2, t));
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			weightM1[i][j] = beta1 * weightM1[i][j] + (1 - beta1) * weightGradient[i][j];
			weightS[i][j] = beta2 * weightS[i][j] + (1 - beta2) * weightGradient[i][j] * weightGradient[i][j];
			weights[i][j] -= params->learningRate * (lambda * weights[i][j] + mScalar * weightM1[i][j] / sqrt(sScalar * weightS[i][j] + 0.0000001));
			weightGradient[i][j] = 0;
		}
	}
}

AdEMAMix::AdEMAMix(double beta1, double beta2, double beta3, double alpha, double lambda) {
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->beta3 = beta3;
	this->alpha = alpha;
	this->lambda = lambda;
}

void AdEMAMix::applyGradient(int m, int n, double** weightM1, double** weightM2, double** weightS, double** weightGradient, double** weights, double t, TrainingParams* params){
	Matrix::add(m, n, weightM1, weightGradient, weightM1, beta1, (1 - beta1));
	Matrix::add(m, n, weightM2, weightGradient, weightM2, beta3, 1 - beta3);
	double mScalar = 1.0 / (1 - pow(beta1, t));
	double sScalar = 1.0 / (1 - pow(beta2, t));
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			weightM1[i][j] = beta1 * weightM1[i][j] + (1 - beta1) * weightGradient[i][j];
			weightS[i][j] = beta2 * weightS[i][j] + (1 - beta2) * weightGradient[i][j] * weightGradient[i][j];
			weights[i][j] -= params->learningRate * (lambda * weights[i][j] + (mScalar * weightM1[i][j] + alpha * weightM2[i][j]) / sqrt(sScalar * weightS[i][j] + 0.0000001));
			weightGradient[i][j] = 0;
		}
	}
}