#include "SequenceMean.h"
#include "Model.h"
#include "ModelParser.h"

const string SequenceMean::LAYER_NAME = "SequenceMean";

SequenceMean::SequenceMean(Activation* activation) {
	this->activation = activation;
	forwardThreadCount = { 0 };
	backThreadCount = { 0 };
	gradientCalculated = { false };
}

SequenceMean::~SequenceMean() {
	delete activation;
	means.free();
	backPropIntermediate.free();
	Layer1D::~Layer1D();
}

void SequenceMean::propagateLayer(int num) {
	Matrix::calculateMean(prevLayer->numTokens[num], size, prevLayer->neurons[num], means, num);
	forwardThreadCount.fetch_add(1);
	if (forwardThreadCount.load() >= batchSize) {
		forwardThreadCount.store(0);
		activation->operate(batchSize, size, means, neurons);
		gradientCalculated.store(false);
		if (nextLayer != NULL) {
			nextLayer->forwardPropagate(num);
		}
	}
}

void SequenceMean::backPropagate(int num) {
	if (num == 0) {
		activation->differentiate(batchSize, size, means, neurons, backPropIntermediate, neuronGradient);
		gradientCalculated.store(true);
	}
	while (!gradientCalculated.load()){}
	float c = 1.0 / prevLayer->numTokens[num];
	for (int i = 0; i < prevLayer->numTokens[num]; i++) {
		for (int j = 0; j < size; j++) {
			prevLayer->neuronGradient[num].r(i, j) = c * backPropIntermediate(num, j);
		}
	}
	prevLayer->backPropagate(num);
}

void SequenceMean::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	size = prevLayer->size;
	prevSize = prevLayer->size + 1;
}

void SequenceMean::setBatchSize(int batchSize) {
	Layer1D::setBatchSize(batchSize);
	means = Matrix(Matrix::ZERO_FILL, batchSize, size, false);
	backPropIntermediate = Matrix(Matrix::ZERO_FILL, batchSize, size, true);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void SequenceMean::save(ofstream& file) {
	file << LAYER_NAME << ",";
	activation->save(file);
	file << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void SequenceMean::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Activation* activation = ModelParser::readActivation(line, commaIndex, newCommaIndex);
	SequenceMean* batchSum = { new SequenceMean(activation) };
	nn->addLayer(batchSum);
}