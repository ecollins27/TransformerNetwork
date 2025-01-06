#include "Dense2D.h"

Dense2D::Dense2D(Activation* activation, int size) {
	this->activation = activation->clone();
	this->size = size;
}

void Dense2D::propagateLayer(int num) {
	Matrix::multiplyABtC(numTokens[num], prevSize, size, prevLayer->neurons[num], weights, linearCombo[num], true);
	activation->operate(numTokens[num], size, linearCombo[num], neurons[num]);
}

void Dense2D::backPropagate(int num) {
	activation->differentiate(numTokens[num], size, linearCombo[num], neurons[num], activationGradient[num]);
	if (activation->isDiagonal()) {
		Matrix::elementMultiply(numTokens[num], size, neuronGradient[num], activationGradientMatrix[num], backPropIntermediate[num], true);
	}
	else {
		Matrix3D::matrixTensorMultiply(numTokens[num], size, size, neuronGradient[num], activationGradient[num], backPropIntermediate[num], true);
	}
	Matrix::multiplyABC(numTokens[num], size, prevSize, backPropIntermediate[num], weights, prevLayer->neuronGradient[num], true);
	Matrix::multiplyAtBC(size, numTokens[num], prevSize, backPropIntermediate[num], prevLayer->neurons[num], weightGradient[num], true);
	optimizer->addGradient(weightGradient[num]);
	prevLayer->backPropagate(num);
}

void Dense2D::setPrevLayer(Layer* prevLayer) {
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
	weights = Matrix(new Matrix::NormalFill(0, stdDeviation), size, prevSize, true);
}

void Dense2D::setBatchSize(int batchSize) {
	Layer2D::initNeurons(batchSize);
	weightGradient = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, size, prevSize, false);
	linearCombo = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, maxNumTokens, size + 1, false);
	backPropIntermediate = Matrix::allocateMatrixArray(Matrix::ZERO_FILL, batchSize, maxNumTokens, size, true);
	if (activation->isDiagonal()) {
		activationGradient = Matrix3D::allocateMatrix3DArray(Matrix::ZERO_FILL, batchSize, 1, maxNumTokens, size);
		activationGradientMatrix = new Matrix[batchSize];
		for (int i = 0; i < batchSize; i++) {
			activationGradientMatrix[i] = Matrix(activationGradient[i].matrix[0], NULL);
		}
	}
	else {
		activationGradient = Matrix3D::allocateMatrix3DArray(Matrix::ZERO_FILL, batchSize, maxNumTokens, size, size);
	}
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void Dense2D::save(ofstream& file) {
	file << "DenseLayer,";
	activation->save(file);
	file << size << ",\n";
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < prevSize; j++) {
			file << weights(i, j) << ",";
		}
		file << "\n";
	}
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void Dense2D::applyGradients(float learningRate, int t) {
	optimizer->applyGradient(weights, t, learningRate, batchSize);
	weights.calculateTranspose(size, prevSize);
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void Dense2D::setOptimizer(Optimizer* optimizer) {
	this->optimizer = optimizer->clone();
	this->optimizer->setDimensions(size, prevSize);
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

int Dense2D::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + size * prevSize;
}