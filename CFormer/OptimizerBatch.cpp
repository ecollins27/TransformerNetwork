#include "OptimizerBatch.h"
#include "MatrixKernel.h"

OptimizerBatch* OptimizerBatch::GRADIENT_DESCENT = { new GradientDescentBatch(0) };
OptimizerBatch* OptimizerBatch::MOMENTUM = { new MomentumBatch(0.9, 0) };
OptimizerBatch* OptimizerBatch::ADAM = { new AdamBatch(0.9,0.999, 0) };
OptimizerBatch* OptimizerBatch::ADEMAMIX = { new AdEMAMixBatch(0.9, 0.9999, 0.999, 5, 0) };

__global__
void kernelCondenseGradients(float*** gradients, float** condensed, int batchSize, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	if (i < N) {
		float sum = 0;
		for (int b = 0; b < batchSize; b++) {
			sum += gradients[b][j][i];
		}
		condensed[j][i] = sum;
	}
}

void OptimizerBatch::condenseGradients() {
	weightGradient.copy(weightGradients[0]);
	for (int i = 1; i < batchSize; i++) {
		MatrixBatch::add(weightGradient, weightGradients[i], weightGradient);
	}
}

void OptimizerBatch::setBatchSize(int batchSize, MatrixBatch* gradients) {
	this->batchSize = batchSize;
	if (gradients == NULL) {
		return;
	}
	weightGradients = gradients;
}

GradientDescentBatch::GradientDescentBatch(float regConstant) {
	this->regConstant = regConstant;
}


void GradientDescentBatch::applyGradient(MatrixBatch& weights, float t, float learningRate) {
	weightGradient.scale(1.0 / batchSize);
	MatrixBatch::linearCombo(1, weights, -learningRate, weightGradient, weights);
	if (regConstant != 0) {
		MatrixBatch::linearCombo(1, weights, -2 * regConstant, weights, weights);
	}
	weightGradient.constantFill(0);
}

OptimizerBatch* GradientDescentBatch::clone() {
	return new GradientDescentBatch(regConstant);
}

void GradientDescentBatch::setDimensions(int height, int width, int depth) {
	this->height = height;
	this->width = width;
	this->depth = depth;
	weightGradient = MatrixBatch(height, width, depth, 0);
}

MomentumBatch::MomentumBatch(float beta, float regConstant) {
	this->beta = beta;
	this->regConstant = regConstant;
}

void MomentumBatch::applyGradient(MatrixBatch& weights, float t, float learningRate) {
	weightGradient.scale(1.0 / batchSize);
	if (regConstant != 0) {
		MatrixBatch::linearCombo(1, weightGradient, 2 * regConstant, weights, weightGradient);
	}
	MatrixBatch::linearCombo(beta, M, -learningRate, weightGradient, M);
	MatrixBatch::add(weights, M, weights);
	weightGradient.constantFill(0);
}

OptimizerBatch* MomentumBatch::clone() {
	return { new MomentumBatch(beta, regConstant) };
}

void MomentumBatch::setDimensions(int height, int width, int depth) {
	this->height = height;
	this->width = width;
	this->depth = depth;
	M = MatrixBatch(height, width, depth, 0);
	weightGradient = MatrixBatch(height, width, depth, 0);
}

AdamBatch::AdamBatch(float beta1, float beta2, float regConstant) {
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->regConstant = regConstant;
}

__global__
void kernelAdam(float** weights, float** M, float** S, float learningRate, float mScalar, float sScalar, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	if (i < N) {
		weights[j][i] -= learningRate * (M[j][i] * mScalar) / sqrt((S[j][i] * sScalar) + 0.0000001);
	}
}

void AdamBatch::applyGradient(MatrixBatch& weights, float t, float learningRate) {
	float mScalar = 1.0 / (1 - pow(beta1, t));
	float sScalar = 1.0 / (1 - pow(beta2, t));
	weightGradient.scale(1.0 / batchSize);
	if (regConstant != 0) {
		MatrixBatch::linearCombo(1, weightGradient, 2 * regConstant, weights, weightGradient);
	}
	MatrixBatch::linearCombo(beta1, M, 1 - beta1, weightGradient, M);
	MatrixBatch::elementMultiply(weightGradient, weightGradient, weightGradient);
	MatrixBatch::linearCombo(beta2, S, 1 - beta2, weightGradient, S);
	weights.copyToDevice(0, 0);
	M.copyToDevice(1, 0);
	S.copyToDevice(2, 0);
	MatrixKernel::runElementKernelBatched(weights.batchSize, weights.height, weights.width, 0, kernelAdam, MatrixBatch::DEVICES[0][0], MatrixBatch::DEVICES[0][1], MatrixBatch::DEVICES[0][2], learningRate, mScalar, sScalar, weights.length);
	weights.copyToHost(0, weights.length, 0);
	weightGradient.constantFill(0);
}

OptimizerBatch* AdamBatch::clone() {
	return { new AdamBatch(beta1, beta2, regConstant) };
}

void AdamBatch::setDimensions(int height, int width, int depth) {
	this->height = height;
	this->width = width;
	this->depth = depth;
	M = MatrixBatch(height, width, depth, 0);
	S = MatrixBatch(height, width, depth, 0);
	weightGradient = MatrixBatch(height, width, depth, 0);
}

AdEMAMixBatch::AdEMAMixBatch(float beta1, float beta2, float beta3, float alpha, float regConstant) {
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->beta3 = beta3;
	this->alpha = alpha;
	this->regConstant = regConstant;
}

__global__
void kernelAdEMAMix(float** weights, float** M1, float** M2, float** S, float learningRate, float mScalar, float sScalar, float alpha, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	if (i < N) {
		weights[j][i] -= learningRate * (mScalar * M1[j][i] + alpha * M2[j][i]) / sqrt(sScalar * S[j][i] + 0.0000001);
	}
}

void AdEMAMixBatch::applyGradient(MatrixBatch& weights, float t, float learningRate) {
	weightGradient.scale(1.0 / batchSize);
	if (regConstant != 0) {
		MatrixBatch::linearCombo(1, weightGradient, 2 * regConstant, weights, weightGradient);
	}
	MatrixBatch::linearCombo(beta1, M1, 1 - beta1, weightGradient, M1);
	MatrixBatch::linearCombo(beta3, M2, 1 - beta3, weightGradient, M2);
	MatrixBatch::elementMultiply(weightGradient, weightGradient, weightGradient);
	MatrixBatch::linearCombo(beta2, S, 1 - beta2, weightGradient, S);
	float mScalar = 1.0 / (1 - pow(beta1, t));
	float sScalar = 1.0 / (1 - pow(beta2, t));
	weights.copyToDevice(0);
	M1.copyToDevice(1, 0);
	M2.copyToDevice(2, 0);
	S.copyToDevice(3, 0);
	MatrixKernel::runElementKernelBatched(weights.batchSize, weights.height, weights.width, 0, kernelAdEMAMix, MatrixBatch::DEVICES[0][0], MatrixBatch::DEVICES[0][1], MatrixBatch::DEVICES[0][2], MatrixBatch::DEVICES[0][3], learningRate, mScalar, sScalar, alpha, weights.length);
	weights.copyToHost(0, weights.length, 0);
	weightGradient.constantFill(0);
}

OptimizerBatch* AdEMAMixBatch::clone() {
	return { new AdEMAMixBatch(beta1, beta2, beta3, alpha, regConstant) };
}

void AdEMAMixBatch::setDimensions(int height, int width, int depth) {
	this->height = height;
	this->width = width;
	this->depth = depth;
	M1 = MatrixBatch(height, width, depth, 0);
	M2 = MatrixBatch(height, width, depth, 0);
	S = MatrixBatch(height, width, depth, 0);
	weightGradient = MatrixBatch(height, width, depth, 0);
}