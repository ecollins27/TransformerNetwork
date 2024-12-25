#include "ResidualAdd2D.h"

ResidualAdd2D::ResidualAdd2D(ResidualSave2D* residualLayer) {
	this->residual = residualLayer;
}

void ResidualAdd2D::propagateLayer(int num) {
	Matrix2::elementAdd(numTokens, size, prevLayer->neurons[num], residual->neurons[num], neurons[num], 1, 1, true);
}

void ResidualAdd2D::backPropagate(int num) {
	neuronGradient[num].copy(numTokens, size, prevLayer->neuronGradient[num]);
	prevLayer->backPropagate(num);
	Matrix2::elementAdd(numTokens, size, residual->neuronGradient[num], neuronGradient[num], residual->neuronGradient[num], 1, 1, true);
	residual->backPropagateWithResidual(num);
}

void ResidualAdd2D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D*>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	size = prevLayer->size;
	prevSize = prevLayer->size + 1;
	if (size != residual->size) {
		throw invalid_argument("Associated ResidualSave2D must have same size");
	}
}

void ResidualAdd2D::save(ofstream& file) {
	file << "ResidualAdd," << residual->index << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}