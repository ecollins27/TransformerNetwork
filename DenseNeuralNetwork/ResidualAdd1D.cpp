#include "ResidualAdd1D.h"

ResidualAdd1D::ResidualAdd1D(ResidualSave1D* residualLayer) {
	residual = residualLayer;
}

void ResidualAdd1D::propagateLayer(int num) {
	Matrix2::elementAdd(batchSize, size, prevLayer->neurons, residual->neurons, neurons, 1, 1, true);
}

void ResidualAdd1D::backPropagate(int num) {
	neuronGradient.copy(batchSize, size, prevLayer->neuronGradient);
	prevLayer->backPropagate(num);
	Matrix2::elementAdd(batchSize, size, neuronGradient, residual->neuronGradient, residual->neuronGradient, 1, 1, true);
	residual->backPropagateWithResidual(num);
}

void ResidualAdd1D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer1D*>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer1D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer1D*)prevLayer;
	this->size = prevLayer->size;
	prevSize = size + 1;
	if (size != residual->size) {
		throw invalid_argument("Associated ResidualSave1D must have same size");
	}
}

void ResidualAdd1D::setBatchSize(int batchSize) {
	Layer1D::setBatchSize(batchSize);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void ResidualAdd1D::save(ofstream& file) {
	file << "ResidualAdd," << residual->index << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}