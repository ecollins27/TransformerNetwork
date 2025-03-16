#include "ResidualAdd1D.h"
#include "Model.h"
#include "ModelParser.h"

const string ResidualAdd1D::LAYER_NAME = "ResidualAdd1D";

ResidualAdd1D::ResidualAdd1D(ResidualSave1D* residualLayer) {
	residual = residualLayer;
}

ResidualAdd1D::~ResidualAdd1D() {
	Layer1D::~Layer1D();
}

void ResidualAdd1D::propagateLayer(int num) {
	Matrix::add(batchSize, size, prevLayer->neurons, residual->neurons, neurons);
}

void ResidualAdd1D::backPropagate(int num) {
	if (num != 0) {
		residual->backPropagateWithResidual(num);
		return;
	}
	neuronGradient.copy(batchSize, size, prevLayer->neuronGradient);
	prevLayer->backPropagate(num);
	Matrix::add(batchSize, size, neuronGradient, residual->neuronGradient, residual->neuronGradient);
	residual->backPropagateWithResidual(num);
}

void ResidualAdd1D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer1D>(prevLayer)) {
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
	file << LAYER_NAME << "," << residual->index << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void ResidualAdd1D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	int saveIndex = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	ResidualAdd1D* residualAdd = { new ResidualAdd1D((ResidualSave1D*)nn->getLayer(saveIndex)) };
	nn->addLayer(residualAdd);
}