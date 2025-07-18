#include "Optimizer.h"
#include "MatrixKernel.h"

Optimizer* Optimizer::GRADIENT_DESCENT = { new GradientDescent(0) };
Optimizer* Optimizer::MOMENTUM = { new Momentum(0.9, 0) };
Optimizer* Optimizer::ADAM = { new Adam(0.9,0.999, 0) };
Optimizer* Optimizer::ADEMAMIX = { new AdEMAMix(0.9, 0.9999, 0.999, 5, 0) };

void Optimizer::condenseGradients() {
	weightGradients.condense(weightGradient);
}

void Optimizer::setBatchSize(int batchSize, Matrix2* gradients) {
	this->batchSize = batchSize;
	if (gradients == NULL) {
		return;
	}
	cudaError_t err = cudaMallocHost(&weightGradients.host, batchSize * sizeof(float*));
	if (err != cudaSuccess) {
		throw runtime_error("CUDA memory allocation failed");
	}
	for (int i = 0; i < batchSize; i++) {
		weightGradients.host[i] = gradients[i].host;
	}
	weightGradients.maxLength = gradients[0].maxLength;
	weightGradients.length = gradients[0].length;
	weightGradients.batchSize = batchSize;
	weightGradients.height = gradients[0].height;
	weightGradients.width = gradients[0].height;
	weightGradients.threadNum = 0;
}

GradientDescent::GradientDescent(float regConstant) {
	this->regConstant = regConstant;
}


void GradientDescent::applyGradient(Matrix2& weights, float t, float learningRate) {
	weightGradient.scale(1.0 / batchSize);
	Matrix2::linearCombo(1, weights, -learningRate, weightGradient, weights);
	if (regConstant != 0) {
		Matrix2::linearCombo(1, weights, -2 * regConstant * learningRate, weights, weights);
	}
	weightGradient.constantFill(0);
}

Optimizer* GradientDescent::clone() {
	return new GradientDescent(regConstant);
}

OptimizerBatch* GradientDescent::cloneBatch() {
	return new GradientDescentBatch(regConstant);
}

void GradientDescent::setDimensions(int height, int width) {
	this->height = height;
	this->width = width;
	weightGradient = Matrix2(height, width, 0);
}

Momentum::Momentum(float beta, float regConstant) {
	this->beta = beta;
	this->regConstant = regConstant;
}

void Momentum::applyGradient(Matrix2& weights, float t, float learningRate) {
	weightGradient.scale(1.0 / batchSize);
	if (regConstant != 0) {
		Matrix2::linearCombo(1, weightGradient, 2 * regConstant, weights, weightGradient);
	}
	Matrix2::linearCombo(beta, M, -learningRate, weightGradient, M);
	Matrix2::add(weights, M, weights);
	weightGradient.constantFill(0);
}

Optimizer* Momentum::clone() {
	return new Momentum(beta, regConstant);
}

OptimizerBatch* Momentum::cloneBatch() {
	return new MomentumBatch(beta, regConstant);
}

void Momentum::setDimensions(int height, int width) {
	this->height = height;
	this->width = width;
	M = Matrix2(height, width, 0);
	weightGradient = Matrix2(height, width, 0);
}

Adam::Adam(float beta1, float beta2, float regConstant) {
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->regConstant = regConstant;
}

__global__
void kernelAdam(float* weights, float* M, float* S, float learningRate, float mScalar, float sScalar, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		weights[i] -= learningRate * (M[i] * mScalar) / sqrt((S[i] * sScalar) + 0.0000001);
	}
}

void Adam::applyGradient(Matrix2& weights, float t, float learningRate) {
	float mScalar = 1.0 / (1 - pow(beta1, t));
	float sScalar = 1.0 / (1 - pow(beta2, t));
	weightGradient.scale(1.0 / batchSize);
	if (regConstant != 0) {
		Matrix2::linearCombo(1, weightGradient, 2 * regConstant, weights, weightGradient);
	}
	Matrix2::linearCombo(beta1, M, 1 - beta1, weightGradient, M);
	Matrix2::elementMultiply(weightGradient, weightGradient, weightGradient);
	Matrix2::linearCombo(beta2, S, 1 - beta2, weightGradient, S);
	weights.copyToDevice(0, 0);
	M.copyToDevice(1, 0);
	S.copyToDevice(2, 0);
	MatrixKernel::runElementKernel(weights.height, weights.width, 0, kernelAdam, Matrix2::DEVICES[0][0], Matrix2::DEVICES[0][1], Matrix2::DEVICES[0][2], learningRate, mScalar, sScalar, weights.length);
	weights.copyToHost(0, weights.length, 0);
	weightGradient.constantFill(0);
}

Optimizer* Adam::clone() {
	return new Adam(beta1, beta2, regConstant);
}

OptimizerBatch* Adam::cloneBatch() {
	return new AdamBatch(beta1, beta2, regConstant);
}

void Adam::setDimensions(int height, int width) {
	this->height = height;
	this->width = width;
	M = Matrix2(height, width, 0);
	S = Matrix2(height, width, 0);
	weightGradient = Matrix2(height, width, 0);
}

AdEMAMix::AdEMAMix(float beta1, float beta2, float beta3, float alpha, float regConstant) {
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->beta3 = beta3;
	this->alpha = alpha;
	this->regConstant = regConstant;
}

__global__
void kernelAdEMAMix(float* weights, float* M1, float* M2, float* S, float learningRate, float mScalar, float sScalar, float alpha, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		weights[i] -= learningRate * (mScalar * M1[i] + alpha * M2[i]) / sqrt(sScalar * S[i] + 0.0000001);
	}
}

void AdEMAMix::applyGradient(Matrix2& weights, float t, float learningRate) {
	weightGradient.scale(1.0 / batchSize);
	if (regConstant != 0) {
		Matrix2::linearCombo(1, weightGradient, 2 * regConstant, weights, weightGradient);
	}
	Matrix2::linearCombo(beta1, M1, 1 - beta1, weightGradient, M1);
	Matrix2::linearCombo(beta3, M2, 1 - beta3, weightGradient, M2);
	Matrix2::elementMultiply(weightGradient, weightGradient, weightGradient);
	Matrix2::linearCombo(beta2, S, 1 - beta2, weightGradient, S);
	float mScalar = 1.0 / (1 - pow(beta1, t));
	float sScalar = 1.0 / (1 - pow(beta2, t));
	weights.copyToDevice(0, 0);
	M1.copyToDevice(1, 0);
	M2.copyToDevice(2, 0);
	S.copyToDevice(3, 0);
	MatrixKernel::runElementKernel(weights.height, weights.width, 0, kernelAdEMAMix, Matrix2::DEVICES[0][0], Matrix2::DEVICES[0][1], Matrix2::DEVICES[0][2], Matrix2::DEVICES[0][3], learningRate, mScalar, sScalar, alpha, weights.length);
	weights.copyToHost(0, weights.length, 0);
	weightGradient.constantFill(0);
}

Optimizer* AdEMAMix::clone(){
	return new AdEMAMix(beta1, beta2, beta3, alpha, regConstant);
}

OptimizerBatch* AdEMAMix::cloneBatch() {
	return new AdEMAMixBatch(beta1, beta2, beta3, alpha, regConstant);
}

void AdEMAMix::setDimensions(int height, int width) {
	this->height = height;
	this->width = width;
	M1 =Matrix2(height, width, 0);
	M2 = Matrix2(height, width, 0);
	S = Matrix2(height, width, 0);
	weightGradient = Matrix2(height, width, 0);
}