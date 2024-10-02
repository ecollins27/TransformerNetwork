#include "DenseLayer.h"

DenseLayer::DenseLayer(Activation* activation, int size) {
	this->activation = activation->clone();
	this->isDiagonal = activation->isDiagonal();
	this->size = size;
	prevLayer = NULL;
	nextLayer = NULL;
}

DenseLayer::~DenseLayer() {
	Matrix::deallocateMatrix(neurons, size + 1, 1);
	Matrix::deallocateMatrix(neuronGradient, size + 1, 1);
	Matrix::deallocateMatrix(weights, size, prevSize);
	Matrix::deallocateMatrix(weightGradient, size, prevSize);
	if (isDiagonal) {
		Matrix::deallocateMatrix(activationGradient[0], batchSize, size);
	} else {
		for (int i = 0; i < this->batchSize; i++) {
			Matrix::deallocateMatrix(activationGradient[i], size, size);
		}
	}
	free(activationGradient);
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
	weightGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, prevSize);
}
void DenseLayer::setNextLayer(Layer* nextLayer) {
	this->nextLayer = nextLayer;
}

void DenseLayer::setBatchSize(int batchSize) {
	if (neurons != NULL) {
		Matrix::deallocateMatrix(neurons, this->batchSize, size + 1);
	} if (neuronGradient != NULL) {
		Matrix::deallocateMatrix(neuronGradient, this->batchSize, size + 1);
	} if (activations != NULL) {
		Matrix::deallocateMatrix(activations, this->batchSize, size + 1);
	} if (backPropIntermediate != NULL) {
		Matrix::deallocateMatrix(backPropIntermediate, this->batchSize, size);
	} if (activationGradient != NULL) {
		if (isDiagonal) {
			Matrix::deallocateMatrix(activationGradient[0], this->batchSize, size);
		} else {
			for (int i = 0; i < this->batchSize; i++) {
				Matrix::deallocateMatrix(activationGradient[i], size, size);
			}
		}
		free(activationGradient);
	}
	this->batchSize = batchSize;
	activation->init(this);
	neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
	activations = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
	for (int i = 0; i < batchSize; i++) {
		neurons[i][size] = 1;
		activations[i][size] = 1;
	}
	neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
	backPropIntermediate = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size);
	if (isDiagonal) {
		activationGradient = (double***)malloc(sizeof(double**));
		activationGradient[0] = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size);
	} else {
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
	Matrix::matrixMultiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights, activations, true);
	Matrix::copy(batchSize, size + 1, activations, neurons);
	activation->operate(this);
	if (nextLayer != NULL) {
		nextLayer->predict();
	}
}

void DenseLayer::forwardPropagate() {
	Matrix::matrixMultiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights, activations, true);
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
	}
	else {
		Matrix::matrixTensorMultiply(batchSize, size, size, neuronGradient, activationGradient, backPropIntermediate, true);
	}
	Matrix::matrixMultiplyABC(batchSize, size, prevSize, backPropIntermediate, weights, prevLayer->neuronGradient, true);
	Matrix::matrixMultiplyAtBC(size, batchSize, prevSize, backPropIntermediate, prevLayer->neurons, weightGradient, true);
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void DenseLayer::applyGradients(double learningRate, int t) {
	Matrix::scale(size, prevSize, weightGradient, 1.0 / batchSize);
	optimizer->applyGradient(weightGradient, weights, t, learningRate);
	activation->applyGradient(this, learningRate, t);
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void DenseLayer::setOptimizer(Optimizer* optimizer) {
	this->optimizer = optimizer->clone();
	this->optimizer->setDimensions(size, prevSize);
	activation->setOptimizer(this, optimizer);
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

void DenseLayer::save(ofstream& file) {
	int index = 0;
	string activationName = typeid(*activation).name();
	for (int i = 0; i < Activation::NUM_ACTIVATIONS; i++) {
		if (activationName.compare(typeid(*(Activation::ALL_ACTIVATIONS[i])).name()) == 0) {
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