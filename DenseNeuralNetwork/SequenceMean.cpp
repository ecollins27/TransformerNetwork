#include "SequenceMean.h"

SequenceMean::SequenceMean(Activation* activation) {
	this->activation = activation;
}

void SequenceMean::propagateLayer(int num) {
	for (int j = 0; j < size; j++) {
		float& mean = means(num, j);
		mean = 0;
		for (int i = 0; i < prevLayer->numTokens[num]; i++) {
			mean += prevLayer->neurons[num](i, j);
		}
		mean /= prevLayer->numTokens[num];
	}
	forwardThreadCount++;
	if (num == batchSize - 1) {
		while (forwardThreadCount != batchSize){}
		activation->operate(batchSize, size, means, neurons);
		forwardThreadCount = 0;
		if (nextLayer != NULL) {
			nextLayer->forwardPropagate(num);
		}
	}
}

void SequenceMean::backPropagate(int num) {
	if (num == 0) {
		activation->differentiate(batchSize, size, means, neurons, activationGradient);
		if (activation->isDiagonal()) {
			Matrix::elementMultiply(batchSize, size, neuronGradient, activationGradientMatrix, backPropIntermediate, true);
		}
		else {
			Matrix3D::matrixTensorMultiply(batchSize, size, size, neuronGradient, activationGradient, backPropIntermediate, true);
		}
		backThreadCount++;
	}
	else {
		while (backThreadCount < 1){}
		backThreadCount++;
	}
	float inverseBatchSize = 1.0 / batchSize;
	for (int i = 0; i < prevLayer->numTokens[num]; i++) {
		for (int j = 0; j < size; j++) {
			prevLayer->neuronGradient[num](i, j) = inverseBatchSize * neuronGradient(num, j);
		}
	}
	if (backThreadCount >= batchSize) {
		backThreadCount = 0;
	}
	prevLayer->backPropagate(num);
}

void SequenceMean::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	size = prevLayer->size;
	prevSize = prevLayer->size + 1;
}

void SequenceMean::setBatchSize(int batchSize) {
	Layer1D::setBatchSize(batchSize);
	means = Matrix(Matrix::ZERO_FILL, batchSize, size + 1, false);
	backPropIntermediate = Matrix(Matrix::ZERO_FILL, batchSize, size, true);
	if (activation->isDiagonal()) {
		activationGradient = Matrix3D(Matrix::ZERO_FILL, 1, batchSize, size);
		activationGradientMatrix = Matrix(activationGradient.matrix[0], NULL);
	}
	else {
		activationGradient = Matrix3D(Matrix::ZERO_FILL, batchSize, size, size);
	}
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void SequenceMean::save(ofstream& file) {
	file << "SequenceMean,";
	activation->save(file);
	file << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}