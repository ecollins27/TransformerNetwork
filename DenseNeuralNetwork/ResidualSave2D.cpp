#include "ResidualSave2D.h"
#include "Model.h"

const string ResidualSave2D::LAYER_NAME = "ResidualSave2D";

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
	file << LAYER_NAME << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void ResidualSave2D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	ResidualSave2D* residualSave = { new ResidualSave2D() };
	nn->addLayer(residualSave);
}

void ResidualSave2D::backPropagateWithResidual(int num) {
	neuronGradient[num].copy(numTokens[num], size, prevLayer->neuronGradient[num]);
	prevLayer->backPropagate(num);
}