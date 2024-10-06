#include "GatedLayer.h"


GatedLayer::GatedLayer(Activation* activation, int size) {
	this->size = size;
	this->activation = activation->clone();
	isDiagonal = this->activation->isDiagonal();
	prevLayer = NULL;
	nextLayer = NULL;
}

template<typename A, typename T>
bool instanceof(T* ptr) {
	return dynamic_cast<A*>(ptr) != NULL;
}

void GatedLayer::setPrevLayer(Layer* prevLayer) {
	this->prevLayer = prevLayer;
	prevSize = prevLayer->size + 1;
	double stdDeviation = sqrt(2.0 / (prevSize + size));
	if (instanceof<Relu>(activation) || instanceof<Elu>(activation)) {
		stdDeviation = sqrt(2.0 / prevSize);
	}
	else if (instanceof<Selu>(activation)) {
		stdDeviation = sqrt(1.0 / prevSize);
	}
	weights1 = Matrix::allocateMatrix({ new Matrix::NormalFill(0, stdDeviation) }, size, prevSize);
	weights2 = Matrix::allocateMatrix({ new Matrix::NormalFill(0, stdDeviation) }, size, prevSize);
	weightGradient1 = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, prevSize);
	weightGradient2 = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, prevSize);
}
void GatedLayer::setNextLayer(Layer* nextLayer) {
	this->nextLayer = nextLayer;
}

void GatedLayer::setBatchSize(int batchSize) {
	if (neurons != NULL) {
		Matrix::deallocateMatrix(neurons, this->batchSize, size + 1);
	} if (neuronGradient != NULL) {
		Matrix::deallocateMatrix(neuronGradient, this->batchSize, size + 1);
	} if (activations1 != NULL) {
		Matrix::deallocateMatrix(activations1, this->batchSize, size + 1);
	} if (backPropIntermediate1 != NULL) {
		Matrix::deallocateMatrix(backPropIntermediate1, this->batchSize, size);
	} if (backPropIntermediate2 != NULL) {
		Matrix::deallocateMatrix(backPropIntermediate2, this->batchSize, size);
	} if (backPropIntermediate3 != NULL) {
		Matrix::deallocateMatrix(backPropIntermediate3, this->batchSize, size);
	} if (activationOutput != NULL) {
		Matrix::deallocateMatrix(activationOutput, this->batchSize, size);
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
	neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
	activations1 = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
	activations2 = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
	for (int i = 0; i < batchSize; i++) {
		neurons[i][size] = 1;
		activations1[i][size] = 1;
		activations2[i][size] = 1;
	}
	neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
	backPropIntermediate1 = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size);
	backPropIntermediate2 = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size);
	backPropIntermediate3 = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size);
	activationOutput = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size);
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

void GatedLayer::predict() {
	Matrix::matrixMultiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights1, activations1, true);
	Matrix::matrixMultiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights2, activations2, true);
	Matrix::copy(batchSize, size + 1, activations1, activationOutput);
	activation->operate(batchSize, size, activations1, activationOutput);
	Matrix::elementMultiply(batchSize, size, activations2, activationOutput, neurons, true);
	if (nextLayer != NULL) {
		nextLayer->predict();
	}
}

void GatedLayer::forwardPropagate() {
	Matrix::matrixMultiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights1, activations1, true);
	Matrix::matrixMultiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights2, activations2, true);
	Matrix::copy(batchSize, size + 1, activations1, activationOutput);
	activation->operate(batchSize, size, activations1, activationOutput);
	Matrix::elementMultiply(batchSize, size, activations2, activationOutput, neurons, true);
	if (nextLayer != NULL) {
		nextLayer->forwardPropagate();
	}
}
void GatedLayer::backPropagate() {
	activation->differentiate(batchSize, size, activations1, activationOutput, activationGradient);
	Matrix::elementMultiply(batchSize, size, neuronGradient, activations2, backPropIntermediate1, true);
	Matrix::elementMultiply(batchSize, size, neuronGradient, activationOutput, backPropIntermediate2, true);
	if (isDiagonal) {
		Matrix::elementMultiply(batchSize, size, backPropIntermediate1, activationGradient[0], backPropIntermediate3, true);
	} else {
		Matrix::matrixTensorMultiply(batchSize, size, size, backPropIntermediate1, activationGradient, backPropIntermediate3, true);
	}
	Matrix::matrixMultiplyAtBC(size, batchSize, prevSize, backPropIntermediate3, prevLayer->neurons, weightGradient1, true);
	Matrix::matrixMultiplyAtBC(size, batchSize, prevSize, backPropIntermediate2, prevLayer->neurons, weightGradient2, true);
	Matrix::matrixMultiplyABC(batchSize, size, prevSize, backPropIntermediate3, weights1, prevLayer->neuronGradient, true);
	Matrix::matrixMultiplyABC(batchSize, size, prevSize, backPropIntermediate2, weights2, prevLayer->neuronGradient, false);
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void GatedLayer::applyGradients(double learningRate, int t) {
	Matrix::scale(size, prevSize, weightGradient1, 1.0 / batchSize);
	Matrix::scale(size, prevSize, weightGradient2, 1.0 / batchSize);
	optimizer->applyGradient(weightGradient1, weights1, t, learningRate);
	optimizer2->applyGradient(weightGradient2, weights2, t, learningRate);
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void GatedLayer::setOptimizer(Optimizer* optimizer) {
	this->optimizer = optimizer->clone();
	this->optimizer->setDimensions(size, prevSize);
	optimizer2 = optimizer->clone();
	optimizer2->setDimensions(size, prevSize);
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

void GatedLayer::save(ofstream& file) {
	file << "GatedLayer,";
	activation->save(file);
	file << size << ",\n";
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < prevSize; j++) {
			file << weights1[i][j] << ",";
		}
		file << "\n";
	}
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < prevSize; j++) {
			file << weights2[i][j] << ",";
		}
		file << "\n";
	}
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

int GatedLayer::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + 2 * size * prevSize;
}