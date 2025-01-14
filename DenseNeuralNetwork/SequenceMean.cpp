#include "SequenceMean.h"

SequenceMean::SequenceMean(Activation* activation) {
	this->activation = activation;
	forwardThreadCount = { 0 };
	backThreadCount = { 0 };
	gradientCalculated = { false };
}

void SequenceMean::propagateLayer(int num) {
	for (int j = 0; j < size; j++) {
		float& mean = means.r(num, j);
		mean = 0;
		for (int i = 0; i < prevLayer->numTokens[num]; i++) {
			mean += prevLayer->neurons[num](i, j);
		}
		mean /= prevLayer->numTokens[num];
	}
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
	file << "SequenceMean,";
	activation->save(file);
	file << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}