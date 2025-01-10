#include "Gated2D.h"

Gated2D::Gated2D(Activation* activation, int size) {
	this->activation = activation->clone();
	this->size = size;
}

void Gated2D::propagateLayer(int num) {
	Matrix::multiplyABtC(numTokens[num], prevSize, size, prevLayer->neurons[num], weights1, A1[num], true);
	Matrix::multiplyABtC(numTokens[num], prevSize, size, prevLayer->neurons[num], weights2, A2[num], true);
	activation->operate(numTokens[num], size, A1[num], Ao[num]);
	Matrix::elementMultiply(numTokens[num], size, Ao[num], A2[num], neurons[num]);
}

void Gated2D::backPropagate(int num) {
	Matrix::elementMultiply(numTokens[num], size, neuronGradient[num], A2[num], A1Grad[num]);
	Matrix::elementMultiply(numTokens[num], size, neuronGradient[num], A1[num], A2Grad[num]);
	activation->differentiate(numTokens[num], size, A1[num], Ao[num], A1Grad[num], AoGrad[num]);
	Matrix::multiplyABC(numTokens[num], size, prevSize, A1Grad[num], weights1, prevLayer->neuronGradient[num], true);
	Matrix::multiplyABC(numTokens[num], size, prevSize, A2Grad[num], weights2, prevLayer->neuronGradient[num], false);
	Matrix::multiplyAtBC(size, numTokens[num], prevSize, A1Grad[num], prevLayer->neurons[num], weightGradient1[num], true);
	Matrix::multiplyAtBC(size, numTokens[num], prevSize, A2Grad[num], prevLayer->neurons[num], weightGradient2[num], true);
	optimizer1->addGradient(weightGradient1[num]);
	optimizer2->addGradient(weightGradient2[num]);
	prevLayer->backPropagate(num);
}

void Gated2D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	prevSize = prevLayer->size + 1;
	float stdDeviation = sqrt(2.0 / (prevSize + size));
	if (instanceOf<Relu>(activation) || instanceOf<Elu>(activation) || instanceOf<Swish>(activation)) {
		stdDeviation = sqrt(2.0 / prevSize);
	}
	else if (instanceOf<Selu>(activation)) {
		stdDeviation = sqrt(1.0 / prevSize);
	}
	weights1 = Matrix(new Matrix::NormalFill(0, stdDeviation), size, prevSize, true);
	weights1.calculateTranspose(size, prevSize);
	weights2 = Matrix(new Matrix::NormalFill(0, stdDeviation), size, prevSize, true);
	weights2.calculateTranspose(size, prevSize);
}

void Gated2D::setBatchSize(int batchSize) {
	Layer2D::initNeurons(batchSize);
	weightGradient1 = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, size, prevSize, false);
	weightGradient2 = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, size, prevSize, false);

	A1 = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, maxNumTokens, size, false);
	A1Grad = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, maxNumTokens, size, true);
	A2 = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, maxNumTokens, size, false);
	A2Grad = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, maxNumTokens, size, true);
	Ao = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, maxNumTokens, size, false);
	Ao = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, maxNumTokens, size, false);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void Gated2D::save(ofstream& file) {
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

void Gated2D::applyGradients(float learningRate, int t) {
	optimizer1->applyGradient(weights1, t, learningRate, batchSize);
	weights1.calculateTranspose(size, prevSize);
	optimizer2->applyGradient(weights2, t, learningRate, batchSize);
	weights2.calculateTranspose(size, prevSize);
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void Gated2D::setOptimizer(Optimizer* optimizer) {
	this->optimizer1 = optimizer->clone();
	this->optimizer1->setDimensions(size, prevSize);
	this->optimizer2 = optimizer->clone();
	this->optimizer2->setDimensions(size, prevSize);
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

int Gated2D::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + 2 * size * prevSize;
}