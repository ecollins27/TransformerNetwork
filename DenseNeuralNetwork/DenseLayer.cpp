#include "DenseLayer.h"

DenseLayer::DenseLayer(Activation* activation, int size) {
	this->activation = activation->clone();
	this->isDiagonal = activation->isDiagonal();
	this->size = size;
	prevLayer = NULL;
	nextLayer = NULL;
}

DenseLayer::~DenseLayer() {
	Matrix::deallocateMatrix(neurons, batchSize, size + 1);
	Matrix::deallocateMatrix(neuronsTranspose, size + 1, batchSize);
	Matrix::deallocateMatrix(neuronGradient, batchSize, size + 1);
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
	float stdDeviation = sqrt(2.0 / (prevSize + size));
	if (instanceof<Relu>(activation) || instanceof<Elu>(activation) || instanceof<Swish>(activation)) {
		stdDeviation = sqrt(2.0 / prevSize);
	} else if (instanceof<Selu>(activation)) {
		stdDeviation = sqrt(1.0 / prevSize);
	}
	weights = Matrix::allocateMatrix({ new Matrix::NormalFill(0,stdDeviation) }, size, prevSize);
	weightsTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, prevSize, size);
	Matrix::transpose(size, prevSize, weights, weightsTranspose);
	weightGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, prevSize);
}

void DenseLayer::setBatchSize(int batchSize) {
	if (maxBatchSize > 0) {
		this->batchSize = batchSize;
	}
	else {
		if (neurons != NULL) {
			Matrix::deallocateMatrix(neurons, this->batchSize, size + 1);
			Matrix::deallocateMatrix(neuronsTranspose, size + 1, this->batchSize);
			Matrix::deallocateMatrix(neuronGradient, this->batchSize, size + 1);
			Matrix::deallocateMatrix(activations, this->batchSize, size + 1);
			Matrix::deallocateMatrix(backPropIntermediate, this->batchSize, size);
			Matrix::deallocateMatrix(backPropIntermediateTranspose, size, this->batchSize);
			Matrix::deallocate3DMatrix(activationGradient, isDiagonal ? 1 : this->batchSize, size, size);
		}
		this->batchSize = batchSize;
		neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
		neuronsTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, batchSize);
		activations = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
		for (int i = 0; i < batchSize; i++) {
			neurons[i][size] = 1;
			neuronsTranspose[size][i] = 1;
			activations[i][size] = 1;
		}
		neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
		backPropIntermediate = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size);
		backPropIntermediateTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, batchSize);
		activationGradient = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, isDiagonal ? 1 : batchSize, size, size);
	}
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void DenseLayer::setMaxBatchSize(int maxBatchSize) {
	if (maxBatchSize > 0) {
		if (neurons != NULL) {
			Matrix::deallocateMatrix(neurons, this->maxBatchSize, size + 1);
			Matrix::deallocateMatrix(neuronsTranspose, size + 1, this->maxBatchSize);
			Matrix::deallocateMatrix(neuronGradient, this->maxBatchSize, size + 1);
			Matrix::deallocateMatrix(activations, this->maxBatchSize, size + 1);
			Matrix::deallocateMatrix(backPropIntermediate, this->maxBatchSize, size);
			Matrix::deallocateMatrix(backPropIntermediateTranspose, size, this->maxBatchSize);
			Matrix::deallocate3DMatrix(activationGradient, isDiagonal ? 1 : this->maxBatchSize, size, size);
		}
		this->maxBatchSize = maxBatchSize;
		neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size + 1);
		neuronsTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, maxBatchSize);
		activations = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size + 1);
		for (int i = 0; i < maxBatchSize; i++) {
			neurons[i][size] = 1;
			neuronsTranspose[size][i] = 1;
			activations[i][size] = 1;
		}
		neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size + 1);
		backPropIntermediate = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size);
		backPropIntermediateTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, maxBatchSize);
		if (isDiagonal) {
			activationGradient = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, 1, maxBatchSize, size);
		}
		else {
			activationGradient = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, maxBatchSize, size, size);
		}
	}
	else {
		this->maxBatchSize = maxBatchSize;
	}
	if (nextLayer != NULL) {
		nextLayer->setMaxBatchSize(maxBatchSize);
	}
}

void DenseLayer::propagateLayer() {
	Matrix::matrixMultiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights, activations, true);
	activation->operate(batchSize, size, activations, neurons);
	Matrix::transpose(batchSize, size, neurons, neuronsTranspose);
}

void DenseLayer::backPropagate() {
	activation->differentiate(batchSize, size, activations, neurons, activationGradient);
	if (isDiagonal) {
		Matrix::elementMultiply(batchSize, size, neuronGradient, activationGradient[0], backPropIntermediate, true);
	}
	else {
		Matrix::matrixTensorMultiply(batchSize, size, size, neuronGradient, activationGradient, backPropIntermediate, true);
	}
	Matrix::transpose(batchSize, size, backPropIntermediate, backPropIntermediateTranspose);
	Matrix::matrixMultiplyABtC(batchSize, size, prevSize, backPropIntermediate, weightsTranspose, prevLayer->neuronGradient, true);
	Matrix::matrixMultiplyABtC(size, batchSize, prevSize, backPropIntermediateTranspose, prevLayer->neuronsTranspose, weightGradient, true);
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void DenseLayer::applyGradients(float learningRate, int t) {
	Matrix::scale(size, prevSize, weightGradient, 1.0 / batchSize);
	optimizer->applyGradient(weightGradient, weights, t, learningRate);
	Matrix::transpose(size, prevSize, weights, weightsTranspose);
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void DenseLayer::setOptimizer(Optimizer* optimizer) {
	this->optimizer = optimizer->clone();
	this->optimizer->setDimensions(size, prevSize);
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

void DenseLayer::save(ofstream& file) {
	file << "DenseLayer,";
	activation->save(file);
	file << size << ",\n";
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

int DenseLayer::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + size * prevSize;
}