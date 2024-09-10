#include "DenseLayer.h"

DenseLayer::DenseLayer(Activation* activation, int size) {
	this->activation = activation;
	this->isDiagonal = activation->isDiagonal();
	this->size = size;
	neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, 1);
	neurons[size][0] = 1;
	neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, 1);
	activationGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, size);
	backPropIntermediate = Matrix::allocateMatrix(Matrix::ZERO_FILL, 1, size);
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

template<typename A, typename T>
bool instanceof(T* ptr) {
	return dynamic_cast<A*>(ptr) != NULL;
}

void DenseLayer::setPrevLayer(Layer* prevLayer) {
	this->prevLayer = prevLayer;
	prevSize = prevLayer->size + 1;
	double stdDeviation = sqrt(2.0 / (prevSize + size));
	if (instanceof<Relu>(activation) || instanceof<Elu>(activation)) {
		stdDeviation = sqrt(2.0 / prevSize);
	} else if (instanceof<Selu>(activation)) {
		stdDeviation = sqrt(1.0 / prevSize);
	}
	weights = Matrix::allocateMatrix({ new Matrix::NormalFill(0,stdDeviation) }, size, prevSize);
	weightM = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, prevSize);
	weightS = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, prevSize);
	weightGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, prevSize);
}
void DenseLayer::setNextLayer(Layer* nextLayer) {
	this->nextLayer = nextLayer;
}

void DenseLayer::forwardPropagate() {
	Matrix::multiplyABC(size, prevSize, 1, weights, prevLayer->neurons, neurons, true);
	activation->operate(this);
	if (nextLayer != NULL) {
		nextLayer->forwardPropagate();
	}
}

void DenseLayer::backPropagate() {
	activation->differentiate(this);
	if (isDiagonal) {
		for (int i = 0; i < size; i++) {
			backPropIntermediate[0][i] = neuronGradient[i][0] * activationGradient[i][i];
		}
	} else {
		Matrix::multiplyAtBC(1, size, size, neuronGradient, activationGradient, backPropIntermediate, true);
	}
	Matrix::multiplyABCt(1, size, prevSize, backPropIntermediate, weights, prevLayer->neuronGradient, true);

	Matrix::multiplyAtBtC(size, 1, prevSize, backPropIntermediate, prevLayer->neurons, weightGradient, false);
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

void DenseLayer::save(ofstream& file) {
	int index = 0;
	for (int i = 0; i < 7; i++) {
		if (activation == Activation::ALL_ACTIVATIONS[i]) {
			index = i;
		}
	}
	file << index << "," << size << "\n";
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < prevSize; j++) {
			file << weights[i][j] << ",";
		}
		file << "\n";
	}
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}