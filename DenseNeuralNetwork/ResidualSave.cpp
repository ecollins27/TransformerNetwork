#include "ResidualSave.h"

void ResidualSave::propagateLayer() {
	Matrix::copy(batchSize, size, prevLayer->neurons, neurons);
	Matrix::transpose(batchSize, size, neurons, neuronsTranspose);
}

void ResidualSave::backPropagate() {
	return;
}

void ResidualSave::backPropagateWithResidual() {
	Matrix::copy(batchSize, size, neuronGradient, prevLayer->neuronGradient);
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void ResidualSave::setPrevLayer(Layer* prevLayer) {
	this->prevLayer = prevLayer;
	this->prevSize = prevLayer->size;
	this->size = prevSize;
}

void ResidualSave::save(ofstream& file) {
	file << "ResidualSave,\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}