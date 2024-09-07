#include "Optimizer.h"

Optimizer* Optimizer::GRADIENT_DESCENT = { new GradientDescent() };
Optimizer* Optimizer::MOMENTUM = { new Momentum(0.9) };
Optimizer* Optimizer::ADAM = { new Adam(0.9,0.999) };

void GradientDescent::applyGradient(int m, int n, double** weightM, double** weightS, double** weightGradient, double** weights, double t, TrainingParams* params) {
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

void Momentum::applyGradient(int m, int n, double** weightM, double** weightS, double** weightGradient, double** weights, double t, TrainingParams* params) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			weightM[i][j] = beta * weightM[i][j] - params->learningRate * weightGradient[i][j];
			weights[i][j] += weightM[i][j];
			weightGradient[i][j] = 0;
		}
	}
}

Adam::Adam(double beta1, double beta2) {
	this->beta1 = beta1;
	this->beta2 = beta2;
}

void Adam::applyGradient(int m, int n, double** weightM, double** weightS, double** weightGradient, double** weights, double t, TrainingParams* params) {
	Matrix::add(m, n, weightM, weightGradient, weightM, beta1, (1 - beta1));
	double mScalar = 1.0 / (1 - pow(beta1, t));
	double sScalar = 1.0 / (1 - pow(beta2, t));
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			weightM[i][j] = beta1 * weightM[i][j] + (1 - beta1) * weightGradient[i][j];
			weightS[i][j] = beta2 * weightS[i][j] + (1 - beta2) * weightGradient[i][j] * weightGradient[i][j];
			weights[i][j] -= params->learningRate * mScalar * weightM[i][j] / sqrt(sScalar * weightS[i][j] + 0.0000001);
			weightGradient[i][j] = 0;
		}
	}
}