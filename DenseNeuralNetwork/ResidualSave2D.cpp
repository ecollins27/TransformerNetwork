#include "ResidualSave2D.h"

void ResidualSave2D::propagateLayer(int num) {
	prevLayer->neurons[num].copy(numTokens[num], size, neurons[num]);
}

void ResidualSave2D::backPropagate(int num) {
	return;
}

void ResidualSave2D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	prevSize = prevLayer->size + 1;
	size = prevLayer->size;
}

void ResidualSave2D::save(ofstream& file) {
	file << "ResidualSave,\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void ResidualSave2D::backPropagateWithResidual(int num) {
	neuronGradient[num].copy(numTokens[num], size, prevLayer->neuronGradient[num]);
	prevLayer->backPropagate(num);
}