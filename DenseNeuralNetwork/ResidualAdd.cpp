#include "ResidualAdd.h"

ResidualAdd::ResidualAdd(ResidualSave* saveLayer) {
	this->saveLayer = saveLayer;
}

void ResidualAdd::propagateLayer() {
	Matrix::copy(batchSize, size, prevLayer->neurons, neurons);
	Matrix::add(batchSize, size, neurons, saveLayer->neurons, neurons, 1, 1);
	Matrix::transpose(batchSize, size, neurons, neuronsTranspose);
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
	this->size = prevLayer->size;
	prevSize = size + 1;
}

void ResidualAdd::save(ofstream& file) {
	file << "ResidualAdd," << saveLayer->getLayerIndex() << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}