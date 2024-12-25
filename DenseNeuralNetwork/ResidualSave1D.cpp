#include "ResidualSave1D.h"

void ResidualSave1D::propagateLayer(int num) {
	prevLayer->neurons.copy(batchSize, size, neurons);
}

void ResidualSave1D::backPropagate(int num) {
	return;
}

void ResidualSave1D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer1D*>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer1D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer1D*)prevLayer;
	this->size = prevLayer->size;
	prevSize = size + 1;
}

void ResidualSave1D::setBatchSize(int batchSize) {
	Layer1D::setBatchSize(batchSize);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void ResidualSave1D::save(ofstream& file) {
	file << "ResidualSave,\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void ResidualSave1D::backPropagateWithResidual(int num) {
	neuronGradient.copy(batchSize, size, prevLayer->neuronGradient);
	prevLayer->backPropagate(num);
}