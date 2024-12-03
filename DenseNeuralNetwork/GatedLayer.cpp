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
	float stdDeviation = 2.0 / (prevSize + size);
	if (instanceof<Relu>(activation) || instanceof<Elu>(activation) || instanceof<Swish>(activation)) {
		stdDeviation = 2.0 / prevSize;
	}
	else if (instanceof<Selu>(activation)) {
		stdDeviation = 1.0 / prevSize;
	}
	weights1 = Matrix::allocateMatrix({ new Matrix::NormalFill(0, stdDeviation) }, size, prevSize);
	weights1Transpose = Matrix::allocateMatrix(Matrix::ZERO_FILL, prevSize, size);
	Matrix::transpose(size, prevSize, weights1, weights1Transpose);
	weights2 = Matrix::allocateMatrix({ new Matrix::NormalFill(0, stdDeviation) }, size, prevSize);
	weights2Transpose = Matrix::allocateMatrix(Matrix::ZERO_FILL, prevSize, size);
	Matrix::transpose(size, prevSize, weights2, weights2Transpose);
	weightGradient1 = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, prevSize);
	weightGradient2 = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, prevSize);
}

void GatedLayer::setBatchSize(int batchSize) {
	if (maxBatchSize > 0) {
		this->batchSize = batchSize;
	}
	else {
		if (neurons != NULL) {
			Matrix::deallocateMatrix(neurons, this->batchSize, size + 1);
			Matrix::deallocateMatrix(neuronsTranspose, size + 1, this->batchSize);
			Matrix::deallocateMatrix(neuronGradient, this->batchSize, size + 1);
			Matrix::deallocateMatrix(activations1, this->batchSize, size + 1);
			Matrix::deallocateMatrix(backPropIntermediate1, this->batchSize, size);
			Matrix::deallocateMatrix(backPropIntermediate2, this->batchSize, size);
			Matrix::deallocateMatrix(backPropIntermediate2Transpose, size, this->batchSize);
			Matrix::deallocateMatrix(backPropIntermediate3, this->batchSize, size);
			Matrix::deallocateMatrix(backPropIntermediate3Transpose, size, this->batchSize);
			Matrix::deallocateMatrix(activationOutput, this->batchSize, size);
			Matrix::deallocate3DMatrix(activationGradient, isDiagonal ? 1 : this->batchSize, size, size);
		}
		this->batchSize = batchSize;
		neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
		neuronsTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, batchSize);
		activations1 = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
		activations2 = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
		neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
		for (int i = 0; i < batchSize; i++) {
			neurons[i][size] = 1;
			neuronGradient[i][size] = 0;
			activations1[i][size] = 1;
			activations2[i][size] = 1;
		}
		backPropIntermediate1 = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size);
		backPropIntermediate2 = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size);
		backPropIntermediate2Transpose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, batchSize);
		backPropIntermediate3 = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size);
		backPropIntermediate3Transpose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, batchSize);
		activationOutput = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size);
		activationGradient = Matrix::allocate3DMatrix(Matrix::ZERO_FILL, isDiagonal ? 1 : batchSize, size, size);
	}
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void GatedLayer::setMaxBatchSize(int batchSize) {
	if (maxBatchSize > 0) {
		if (neurons != NULL) {
			Matrix::deallocateMatrix(neurons, this->maxBatchSize, size + 1);
			Matrix::deallocateMatrix(neuronsTranspose, size + 1, this->maxBatchSize);
			Matrix::deallocateMatrix(neuronGradient, this->maxBatchSize, size + 1);
			Matrix::deallocateMatrix(activations1, this->maxBatchSize, size + 1);
			Matrix::deallocateMatrix(backPropIntermediate1, this->maxBatchSize, size);
			Matrix::deallocateMatrix(backPropIntermediate2, this->maxBatchSize, size);
			Matrix::deallocateMatrix(backPropIntermediate2Transpose, size, this->maxBatchSize);
			Matrix::deallocateMatrix(backPropIntermediate3, this->maxBatchSize, size);
			Matrix::deallocateMatrix(backPropIntermediate3Transpose, size, this->maxBatchSize);
			Matrix::deallocateMatrix(activationOutput, this->maxBatchSize, size);
			Matrix::deallocate3DMatrix(activationGradient, isDiagonal ? 1 : this->maxBatchSize, size, size);
		}
		this->maxBatchSize = maxBatchSize;
		neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size + 1);
		neuronsTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, maxBatchSize);
		activations1 = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size + 1);
		activations2 = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size + 1);
		neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size + 1);
		for (int i = 0; i < maxBatchSize; i++) {
			neurons[i][size] = 1;
			neuronGradient[i][size] = 0;
			activations1[i][size] = 1;
			activations2[i][size] = 1;
		}
		backPropIntermediate1 = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size);
		backPropIntermediate2 = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size);
		backPropIntermediate2Transpose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, maxBatchSize);
		backPropIntermediate3 = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size);
		backPropIntermediate3Transpose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size, maxBatchSize);
		activationOutput = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size);
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

void GatedLayer::propagateLayer() {
	Matrix::matrixMultiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights1, activations1, true);
	Matrix::matrixMultiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights2, activations2, true);
	activation->operate(batchSize, size, activations1, activationOutput);
	Matrix::elementMultiply(batchSize, size, activations2, activationOutput, neurons, true);
	Matrix::transpose(batchSize, size + 1, neurons, neuronsTranspose);
}
void GatedLayer::backPropagate() {
	activation->differentiate(batchSize, size, activations1, activationOutput, activationGradient);
	Matrix::elementMultiply(batchSize, size, neuronGradient, activations2, backPropIntermediate1, true);
	Matrix::elementMultiply(batchSize, size, neuronGradient, activationOutput, backPropIntermediate2, true);
	Matrix::transpose(batchSize, size, backPropIntermediate2, backPropIntermediate2Transpose);
	if (isDiagonal) {
		Matrix::elementMultiply(batchSize, size, backPropIntermediate1, activationGradient[0], backPropIntermediate3, true);
	} else {
		Matrix::matrixTensorMultiply(batchSize, size, size, backPropIntermediate1, activationGradient, backPropIntermediate3, true);
	}
	Matrix::transpose(batchSize, size, backPropIntermediate3, backPropIntermediate3Transpose);
	Matrix::matrixMultiplyABtC(size, batchSize, prevSize, backPropIntermediate3Transpose, prevLayer->neuronsTranspose, weightGradient1, true);
	Matrix::matrixMultiplyABtC(size, batchSize, prevSize, backPropIntermediate2Transpose, prevLayer->neuronsTranspose, weightGradient2, true);
	Matrix::matrixMultiplyABtC(batchSize, size, prevSize, backPropIntermediate3, weights1Transpose, prevLayer->neuronGradient, true);
	Matrix::matrixMultiplyABtC(batchSize, size, prevSize, backPropIntermediate2, weights2Transpose, prevLayer->neuronGradient, false);
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void GatedLayer::applyGradients(float learningRate, int t) {
	Matrix::scale(size, prevSize, weightGradient1, 1.0 / batchSize);
	Matrix::scale(size, prevSize, weightGradient2, 1.0 / batchSize);
	optimizer->applyGradient(weightGradient1, weights1, t, learningRate);
	optimizer2->applyGradient(weightGradient2, weights2, t, learningRate);
	Matrix::transpose(size, prevSize, weights1, weights1Transpose);
	Matrix::transpose(size, prevSize, weights2, weights2Transpose);
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