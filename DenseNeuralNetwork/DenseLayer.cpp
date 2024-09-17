#include "DenseLayer.h"

DenseLayer::DenseLayer(Activation* activation, int size) {
	this->activation = activation;
	this->isDiagonal = activation->isDiagonal();
	this->size = size;
	prevLayer = NULL;
	nextLayer = NULL;
	if (activation == Activation::ELU) {
		printf("%d true\n", size);
	}
}

DenseLayer::~DenseLayer() {
	Matrix::deallocateMatrix(neurons, size + 1, 1);
	Matrix::deallocateMatrix(neuronGradient, size + 1, 1);
	Matrix::deallocateMatrix(weights, size, prevSize);
	Matrix::deallocateMatrix(weightGradient, size, prevSize);
	Matrix::deallocateMatrix(weightM1, size, prevSize);
	Matrix::deallocateMatrix(weightM2, size, prevSize);
	Matrix::deallocateMatrix(weightS, size, prevSize);
	if (isDiagonal) {
		Matrix::deallocateMatrix(activationGradient[0], size, size);
		free(activationGradient);
	}
	else {
		for (int i = 0; i < batchSize; i++) {
			Matrix::deallocateMatrix(activationGradient[i], size, size);
		}
		free(activationGradient);
	}
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
	weightM1 = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, prevSize);
	weightM2 = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, prevSize);
	weightS = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, prevSize);
	weightGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, prevSize);
}
void DenseLayer::setNextLayer(Layer* nextLayer) {
	this->nextLayer = nextLayer;
}

void DenseLayer::setBatchSize(int batchSize) {
	this->batchSize = batchSize;
	neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
	for (int i = 0; i < batchSize; i++) {
		neurons[i][size] = 1;
	}
	neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
	activations = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
	backPropIntermediate = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size);
	if (isDiagonal) {
		activationGradient = (double***)malloc(sizeof(double**));
		activationGradient[0] = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size);
	}
	else {
		activationGradient = (double***)malloc(batchSize * sizeof(double**));
		for (int i = 0; i < batchSize; i++) {
			activationGradient[i] = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, size);
		}
	}
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void DenseLayer::predict() {
	Matrix::multiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights, activations, true);
	Matrix::copy(batchSize, size + 1, activations, neurons);
	activation->operate(this);
	if (nextLayer != NULL) {
		nextLayer->predict();
	}
}

void DenseLayer::forwardPropagate() {
	Matrix::multiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights, activations, true);
	Matrix::copy(batchSize, size + 1, activations, neurons);
	activation->operate(this);
	if (nextLayer != NULL) {
		nextLayer->forwardPropagate();
	}
}

void DenseLayer::backPropagate() {
	activation->differentiate(this);
	if (isDiagonal) {
		Matrix::elementMultiply(batchSize, size, neuronGradient, activationGradient[0], backPropIntermediate, true);
	} else {
		Matrix::matrixTensorMultiply(batchSize, size, size, neuronGradient, activationGradient, backPropIntermediate, true);
	}
	Matrix::multiplyABC(batchSize, size, prevSize, backPropIntermediate, weights, prevLayer->neuronGradient, true);
	Matrix::multiplyAtBC(size, batchSize, prevSize, backPropIntermediate, prevLayer->neurons, weightGradient, true);
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void DenseLayer::applyGradients(TrainingParams* params, int t) {
	Matrix::scale(size, prevSize, weightGradient, 1.0 / params->batchSize);
	params->optimizer->applyGradient(size, prevSize, weightM1, weightM2, weightS, weightGradient, weights, t, params);
	if (nextLayer != NULL) {
		nextLayer->applyGradients(params, t);
	}
}

void DenseLayer::save(ofstream& file) {
	int index = 0;
	for (int i = 0; i < 8; i++) {
		if (activation == Activation::ALL_ACTIVATIONS[i]) {
			index = i;
		}
	}
	file << "DenseLayer," << index << "," << size << "\n";
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