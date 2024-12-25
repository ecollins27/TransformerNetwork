#include "Dense1D.h"

Dense1D::Dense1D(Activation* activation, int size) {
	this->activation = activation->clone();
	this->size = size;
}

void Dense1D::propagateLayer(int num) {
	Matrix2::multiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights, linearCombo, true);
	activation->operate(batchSize, size, linearCombo, neurons);
}

void Dense1D::backPropagate(int num) {
	activation->differentiate(batchSize, size, linearCombo, neurons, activationGradient);
	if (activation->isDiagonal()) {
		Matrix2::elementMultiply(batchSize, size, neuronGradient, activationGradientMatrix, backPropIntermediate, true);
	}
	else {
		Matrix3D::matrixTensorMultiply(batchSize, size, size, neuronGradient, activationGradient, backPropIntermediate, true);
	}
	Matrix2::multiplyABC(batchSize, size, prevSize, backPropIntermediate, weights, prevLayer->neuronGradient, true);
	backPropIntermediate.calculateTranspose(batchSize, size);
	prevLayer->neurons.calculateTranspose(batchSize, prevSize - 1);
	Matrix2::multiplyAtBC(size, batchSize, prevSize, backPropIntermediate, prevLayer->neurons, weightGradient, true);
	prevLayer->backPropagate(num);
}

void Dense1D::setPrevLayer(Layer* prevLayer) {
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
	weights = Matrix2(new Matrix2::NormalFill(0, stdDeviation), size, prevSize, true);
}

void Dense1D::setBatchSize(int batchSize) {
	Layer1D::setBatchSize(batchSize);
	linearCombo = Matrix2(Matrix2::ZERO_FILL, batchSize, size + 1, false);
	backPropIntermediate = Matrix2(Matrix2::ZERO_FILL, batchSize, size, true);
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

void Dense1D::save(ofstream& file) {
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

void Dense1D::applyGradients(float learningRate, int t) {
	optimizer->applyGradient(weights, t, learningRate, batchSize);
	weights.calculateTranspose(size, prevSize);
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void Dense1D::setOptimizer(Optimizer* optimizer) {
	this->optimizer = optimizer->clone();
	this->optimizer->setDimensions(size, prevSize);
	weightGradient = this->optimizer->weightGradient;
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

int Dense1D::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + size * prevSize;
}