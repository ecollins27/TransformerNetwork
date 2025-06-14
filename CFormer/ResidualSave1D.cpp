#include "ResidualSave1D.h"
#include "Model.h"
#include "ModelParser.h"

const string ResidualSave1D::LAYER_NAME = "ResidualSave1D";

ResidualSave1D::~ResidualSave1D() {
	Layer1D::~Layer1D();
}

void ResidualSave1D::propagateLayer(int num) {
	prevLayer->neurons.copy(batchSize, size, neurons);
}

void ResidualSave1D::backPropagate(int num) {
	return;
}

void ResidualSave1D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer1D>(prevLayer)) {
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
	file << LAYER_NAME << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void ResidualSave1D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	ResidualSave1D* residualSave = { new ResidualSave1D() };
	nn->addLayer(residualSave);
}

void ResidualSave1D::backPropagateWithResidual(int num) {
	if (num != 0) {
		prevLayer->backPropagate(num);
		return;
	}
	neuronGradient.copy(batchSize, size, prevLayer->neuronGradient);
	prevLayer->backPropagate(num);
}