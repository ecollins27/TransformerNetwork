#include "ResidualAdd2D.h"
#include "Model.h"
#include "ModelParser.h"

const string ResidualAdd2D::LAYER_NAME = "ResidualAdd2D";

ResidualAdd2D::ResidualAdd2D(ResidualSave2D* residualLayer) {
	this->residual = residualLayer;
}

void ResidualAdd2D::propagateLayer(int num) {
	Matrix::add(numTokens[num], size, prevLayer->neurons[num], residual->neurons[num], neurons[num]);
}

void ResidualAdd2D::backPropagate(int num) {
	neuronGradient[num].copy(numTokens[num], size, prevLayer->neuronGradient[num]);
	prevLayer->backPropagate(num);
	Matrix::add(numTokens[num], size, residual->neuronGradient[num], neuronGradient[num], residual->neuronGradient[num]);
	residual->backPropagateWithResidual(num);
}

void ResidualAdd2D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
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
	file << LAYER_NAME << "," << residual->index << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void ResidualAdd2D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	int saveIndex = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	ResidualAdd2D* residualAdd = { new ResidualAdd2D((ResidualSave2D*)nn->getLayer(saveIndex)) };
	nn->addLayer(residualAdd);
}