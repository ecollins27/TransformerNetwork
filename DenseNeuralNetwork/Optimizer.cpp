#include "Optimizer.h"

Optimizer* Optimizer::GRADIENT_DESCENT = { new GradientDescent(0) };
Optimizer* Optimizer::MOMENTUM = { new Momentum(0.9, 0) };
Optimizer* Optimizer::ADAM = { new Adam(0.9,0.999, 0) };
Optimizer* Optimizer::ADEMAMIX = { new AdEMAMix(0.9, 0.9999, 0.999, 5, 0) };

void Optimizer::addGradient(Matrix gradient) {
	Matrix::add(height, width, weightGradient, gradient, weightGradient);
}

GradientDescent::GradientDescent(float regConstant) {
	this->regConstant = regConstant;
}

void GradientDescent::applyGradient(Matrix weights, float t, float learningRate, int batchSize) {
	weightGradient.scale(height, width, 1.0 / batchSize);
	Matrix::linearCombo(height, width, 1, weights, -learningRate, weightGradient, weights);
	if (regConstant != 0) {
		Matrix::linearCombo(height, width, 1, weights, -2 * regConstant, weights, weights);
	}
	weightGradient.constantFill(0, height, width);
	if (weights.saveTranspose) {
		*weights.transposeUpdated = false;
	}
}

Optimizer* GradientDescent::clone() {
	return { new GradientDescent(regConstant) };
}

void GradientDescent::setDimensions(int height, int width) {
	this->height = height;
	this->width = width;
	weightGradient = Matrix(Matrix::ZERO_FILL, height, width, false);
}

Momentum::Momentum(float beta, float regConstant) {
	this->beta = beta;
	this->regConstant = regConstant;
}

void Momentum::applyGradient(Matrix weights, float t, float learningRate, int batchSize) {
	weightGradient.scale(height, width, 1.0 / batchSize);
	Matrix::linearCombo(height, width, beta, M, -learningRate, weightGradient, M);
	if (regConstant != 0) {
		Matrix::linearCombo(height, width, 1, M, 2 * regConstant, weights, M);
	}
	Matrix::add(height, width, weights, M, weights);
	weightGradient.constantFill(0, height, width);
	if (weights.saveTranspose) {
		*weights.transposeUpdated = false;
	}
}

Optimizer* Momentum::clone() {
	return { new Momentum(beta, regConstant) };
}

void Momentum::setDimensions(int height, int width) {
	this->height = height;
	this->width = width;
	M = Matrix(Matrix::ZERO_FILL, height, width, false);
	weightGradient = Matrix(Matrix::ZERO_FILL, height, width, false);
}

Adam::Adam(float beta1, float beta2, float regConstant) {
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->regConstant = regConstant;
}

void Adam::applyGradient(Matrix weights, float t, float learningRate, int batchSize) {
	float mScalar = 1.0 / (1 - pow(beta1, t));
	float sScalar = 1.0 / (1 - pow(beta2, t));
	weightGradient.scale(height, width, 1.0 / batchSize);
	if (regConstant != 0) {
		Matrix::linearCombo(height, width, 1, weightGradient, 2 * regConstant, weights, weightGradient);
	}
	Matrix::linearCombo(height, width, beta1, M, 1 - beta1, weightGradient, M);
	Matrix::elementMultiply(height, width, weightGradient, weightGradient, weightGradient);
	Matrix::linearCombo(height, width, beta2, S, 1 - beta2, weightGradient, S);

	int w4 = width >> 2 << 2;
	__m128 n, d, v;
	__m128 numScalar = _mm_set1_ps(learningRate * mScalar);
	__m128 denScalar = _mm_set1_ps(sScalar);
	__m128 e = _mm_set1_ps(0.0000001);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < w4; j+=4) {
			n = _mm_mul_ps(numScalar, _mm_loadu_ps(&M.matrix[i][j]));
			d = _mm_sqrt_ps(_mm_add_ps(e, _mm_mul_ps(denScalar, _mm_loadu_ps(&S.matrix[i][j]))));
			v = _mm_sub_ps(_mm_loadu_ps(&weights.matrix[i][j]), _mm_div_ps(n, d));
			_mm_store_ps(&weights.matrix[i][j], v);
		}
		for (int j = w4; j < width; j++) {
			weights.r(i, j) -= learningRate * mScalar * M(i, j) / sqrt(sScalar * S(i, j) + 0.0000001);
		}
	}
	weightGradient.constantFill(0, height, width);
	if (weights.saveTranspose) {
		*weights.transposeUpdated = false;
	}
}

Optimizer* Adam::clone() {
	return { new Adam(beta1, beta2, regConstant) };
}

void Adam::setDimensions(int height, int width) {
	this->height = height;
	this->width = width;
	M = Matrix(Matrix::ZERO_FILL, height, width, false);
	S = Matrix(Matrix::ZERO_FILL, height, width, false);
	weightGradient = Matrix(Matrix::ZERO_FILL, height, width, false);
}

AdEMAMix::AdEMAMix(float beta1, float beta2, float beta3, float alpha, float regConstant) {
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->beta3 = beta3;
	this->alpha = alpha;
	this->regConstant = regConstant;
}

void AdEMAMix::applyGradient(Matrix weights, float t, float learningRate, int batchSize) {
	weightGradient.scale(height, width, 1.0 / batchSize);
	if (regConstant != 0) {
		Matrix::linearCombo(height, width, 1, weightGradient, 2 * regConstant, weights, weightGradient);
	}
	Matrix::linearCombo(height, width, beta1, M1, 1 - beta1, weightGradient, M1);
	Matrix::linearCombo(height, width, beta3, M2, 1 - beta3, weightGradient, M2);
	Matrix::elementMultiply(height, width, weightGradient, weightGradient, weightGradient);
	Matrix::linearCombo(height, width, beta2, S, 1 - beta2, weightGradient, S);

	float mScalar = 1.0 / (1 - pow(beta1, t));
	float sScalar = 1.0 / (1 - pow(beta2, t));
	int w4 = width >> 2 << 2;
	__m128 n, d, v;
	__m128 mScal = _mm_set1_ps(learningRate * mScalar);
	__m128 sScal = _mm_set1_ps(sScalar);
	__m128 a = _mm_set1_ps(learningRate * alpha);
	__m128 e = _mm_set1_ps(0.0000001);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < w4; j += 4) {
			n = _mm_add_ps(_mm_mul_ps(mScal, _mm_loadu_ps(&M1.matrix[i][j])), _mm_mul_ps(a, _mm_loadu_ps(&M2.matrix[i][j])));
			d = _mm_sqrt_ps(_mm_add_ps(e, _mm_mul_ps(sScal, _mm_loadu_ps(&S.matrix[i][j]))));
			v = _mm_sub_ps(_mm_loadu_ps(&weights.matrix[i][j]), _mm_div_ps(n, d));
			_mm_store_ps(&weights.matrix[i][j], v);
		}
		for (int j = w4; j < width; j++) {
			weights.r(i, j) -= learningRate * (mScalar * M1(i, j) + alpha * M2(i, j)) / sqrt(sScalar * S(i, j) + 0.0000001);
		}
	}
	weightGradient.constantFill(0, height, width);
	if (weights.saveTranspose) {
		*weights.transposeUpdated = false;
	}

	//float mScalar = 1.0 / (1 - pow(beta1, t));
	//float sScalar = 1.0 / (1 - pow(beta2, t));
	//float fullGradient;
	//for (int i = 0; i < height; i++) {
	//	for (int j = 0; j < width; j++) {
	//		weightGradient.r(i, j) /= batchSize;
	//		fullGradient = weightGradient(i, j) + 2 * regConstant * weights(i, j);
	//		M1.r(i, j) = beta1 * M1(i, j) + (1 - beta1) * fullGradient;
	//		M2.r(i, j) = beta3 * M2(i, j) + (1 - beta3) * fullGradient;
	//		S.r(i, j) = beta2 * S(i, j) + (1 - beta2) * fullGradient * fullGradient;
	//		weights.r(i, j) -= learningRate * (mScalar * M1(i, j) + alpha * M2(i, j)) / sqrt(sScalar * S(i, j) + 0.0000001);
	//		weightGradient.r(i, j) = 0;
	//	}
	//}
}

Optimizer* AdEMAMix::clone(){
	return { new AdEMAMix(beta1, beta2, beta3, alpha, regConstant) };
}

void AdEMAMix::setDimensions(int height, int width) {
	this->height = height;
	this->width = width;
	M1 =Matrix(Matrix::ZERO_FILL, height, width, false);
	M2 = Matrix(Matrix::ZERO_FILL, height, width, false);
	S = Matrix(Matrix::ZERO_FILL, height, width, false);
	weightGradient = Matrix(Matrix::ZERO_FILL, height, width, false);
}