#include "ResidualAdd.h"

ResidualAdd::ResidualAdd(ResidualSave* saveLayer) {
	this->saveLayer = saveLayer;
}

void ResidualAdd::predict() {
	Matrix::copy(batchSize, size, prevLayer->neurons, neurons);
	Matrix::add(batchSize, size, neurons, saveLayer->neurons, neurons, 1, 1);
	if (nextLayer != NULL) {
		nextLayer->predict();
	}
}

void ResidualAdd::forwardPropagate() {
	Matrix::copy(batchSize, size, prevLayer->neurons, neurons);
	Matrix::add(batchSize, size, neurons, saveLayer->neurons, neurons, 1, 1);
	Matrix::transpose(batchSize, size, neurons, neuronsTranspose);
	if (nextLayer != NULL) {
		nextLayer->predict();
	}
}

void ResidualAdd::backPropagate() {
	Matrix::copy(batchSize, size, neuronGradient, prevLayer->neuronGradient);
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
	Matrix::add(batchSize, size, neuronGradient, saveLayer->neuronGradient, saveLayer->neuronGradient, 1, 1);
	saveLayer->backPropagateWithResidual();
}

void ResidualAdd::setPrevLayer(Layer* prevLayer) {
	this->prevLayer = prevLayer;
	this->prevSize = prevLayer->size;
	this->size = prevSize;
}

void ResidualAdd::save(ofstream& file) {
	file << "ResidualAdd," << saveLayer->getLayerIndex() << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}