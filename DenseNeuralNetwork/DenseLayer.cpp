#include "DenseLayer.h"

DenseLayer::DenseLayer(Activation* activation, int size) {
	this->activation = activation;
	this->isDiagonal = activation->isDiagonal();
	this->size = size;
	neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, 1);
	neurons[size][0] = 1;
	neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, 1);
	activationGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, size);
	prevLayer = NULL;
	nextLayer = NULL;
}

DenseLayer::~DenseLayer() {
	Matrix::deallocateMatrix(neurons, size + 1, 1);
	Matrix::deallocateMatrix(neuronGradient, size + 1, 1);
	Matrix::deallocateMatrix(weights, size, prevSize);
	Matrix::deallocateMatrix(weightGradient, size, prevSize);
	Matrix::deallocateMatrix(activationGradient, size, size);
}

void DenseLayer::setPrevLayer(Layer* prevLayer) {
	this->prevLayer = prevLayer;
	prevSize = prevLayer->size + 1;
	weights = Matrix::allocateMatrix({ new Matrix::NormalFill(0,sqrt(1.0 / prevSize)) }, size, prevSize);
	weightM = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, prevSize);
	weightS = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, prevSize);
	weightGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, prevSize);
}
void DenseLayer::setNextLayer(Layer* nextLayer) {
	this->nextLayer = nextLayer;
}

void DenseLayer::forwardPropagate() {
	Matrix::multiply(size, prevSize, 1, weights, prevLayer->neurons, neurons);
	activation->operate(this);
	if (nextLayer != NULL) {
		nextLayer->forwardPropagate();
	}
}
void DenseLayer::backPropagate() {
	activation->differentiate(this);
	for (int i = 0; i < prevSize; i++) {
		double& prevNeuronGradient = prevLayer->neuronGradient[i][0];
		prevNeuronGradient = 0;
		double prevNeuron = prevLayer->neurons[i][0];
		for (int j = 0; j < size; j++) {
			double dLdz = neuronGradient[j][0];
			int k = isDiagonal ? j : 0;
			do {
				prevNeuronGradient += dLdz * activationGradient[j][k] * weights[k][i];
				weightGradient[j][i] += prevNeuron * neuronGradient[k][0] * activationGradient[k][j];
				k++;
			} while (k < size && !isDiagonal);
		}
	}
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void DenseLayer::applyGradients(TrainingParams* params, int t) {
	Matrix::scale(size, prevSize, weightGradient, 1.0 / params->batchSize);
	params->optimizer->applyGradient(size, prevSize, weightM, weightS, weightGradient, weights, t, params);
	if (nextLayer != NULL) {
		nextLayer->applyGradients(params, t);
	}
}