#include "Gated1D.h"

Gated1D::Gated1D(Activation* activation, int size) {
	this->activation = activation->clone();
	this->size = size;
}

void Gated1D::propagateLayer(int num) {
	Matrix2::multiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights1, A1, true);
	Matrix2::multiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights2, A2, true);
	activation->operate(batchSize, size, A1, Ao);
	Matrix2::elementMultiply(batchSize, size, Ao, A2, neurons, true);
}

void Gated1D::backPropagate(int num) {
	Matrix2::elementMultiply(batchSize, size, neuronGradient, A2, A1Grad, true);
	Matrix2::elementMultiply(batchSize, size, neuronGradient, A1, A2Grad, true);
	activation->differentiate(batchSize, size, A1, Ao, activationGradient);
	if (activation->isDiagonal()) {
		Matrix2::elementMultiply(batchSize, size, AoGrad, activationGradientMatrix, A1Grad, true);
	}
	else {
		Matrix3D::matrixTensorMultiply(batchSize, size, size, AoGrad, activationGradient, A1Grad, true);
	}
	Matrix2::multiplyABC(batchSize, size, prevSize, A1Grad, weights1, prevLayer->neuronGradient, true);
	Matrix2::multiplyABC(batchSize, size, prevSize, A2Grad, weights2, prevLayer->neuronGradient, false);
	A1Grad.calculateTranspose(batchSize, size);
	A2Grad.calculateTranspose(batchSize, size);
	prevLayer->neurons.calculateTranspose(batchSize, prevSize);
	Matrix2::multiplyAtBC(size, batchSize, prevSize, A1Grad, prevLayer->neurons, weightGradient1, true);
	Matrix2::multiplyAtBC(size, batchSize, prevSize, A2Grad, prevLayer->neurons, weightGradient2, true);
	prevLayer->backPropagate(num);
}

void Gated1D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer1D*>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer1D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer1D*)prevLayer;
	prevSize = prevLayer->size + 1;
	float stdDeviation = sqrt(2.0 / (prevSize + size));
	if (instanceOf<Relu>(activation) || instanceOf<Elu>(activation) || instanceOf<Swish>(activation)) {
		stdDeviation = sqrt(2.0 / prevSize);
	}
	else if (instanceOf<Selu>(activation)) {
		stdDeviation = sqrt(1.0 / prevSize);
	}
	weights1 = Matrix2(new Matrix2::NormalFill(0, stdDeviation), size, prevSize, true);
	weights1.calculateTranspose(size, prevSize);
	weights2 = Matrix2(new Matrix2::NormalFill(0, stdDeviation), size, prevSize, true);
	weights2.calculateTranspose(size, prevSize);
}

void Gated1D::setBatchSize(int batchSize) {
	Layer1D::setBatchSize(batchSize);
	A1 = Matrix2(Matrix2::ZERO_FILL, batchSize, size, false);
	A1Grad = Matrix2(Matrix2::ZERO_FILL, batchSize, size, true);
	A2 = Matrix2(Matrix2::ZERO_FILL, batchSize, size, false);
	A2Grad = Matrix2(Matrix2::ZERO_FILL, batchSize, size, true);
	Ao = Matrix2(Matrix2::ZERO_FILL, batchSize, size, false);
	AoGrad = Matrix2(Matrix2::ZERO_FILL, batchSize, size, false);
	if (activation->isDiagonal()) {
		activationGradient = Matrix3D(Matrix2::ZERO_FILL, 1, batchSize, size);
		activationGradientMatrix = Matrix2(activationGradient.matrix[0], NULL);
	}
	else {
		activationGradient = Matrix3D(Matrix2::ZERO_FILL, batchSize, size, size);
	}
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void Gated1D::save(ofstream& file) {
	file << "GatedLayer,";
	activation->save(file);
	file << size << ",\n";
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < prevSize; j++) {
			file << weights1(i, j) << ",";
		}
		file << "\n";
	}
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < prevSize; j++) {
			file << weights2(i, j) << ",";
		}
		file << "\n";
	}
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void Gated1D::applyGradients(float learningRate, int t) {
	optimizer1->applyGradient(weights1, t, learningRate, batchSize);
	weights1.calculateTranspose(size, prevSize);
	optimizer2->applyGradient(weights2, t, learningRate, batchSize);
	weights2.calculateTranspose(size, prevSize);
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void Gated1D::setOptimizer(Optimizer* optimizer) {
	this->optimizer1 = optimizer->clone();
	this->optimizer1->setDimensions(size, prevSize);
	weightGradient1 = this->optimizer1->weightGradient;
	this->optimizer2 = optimizer->clone();
	this->optimizer2->setDimensions(size, prevSize);
	weightGradient2 = this->optimizer2->weightGradient;
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

int Gated1D::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + 2 * size * prevSize;
}