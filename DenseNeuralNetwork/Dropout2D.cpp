#include "Dropout2D.h"

Dropout2D::Dropout2D(float dropoutRate) {
	this->dropoutRate = dropoutRate;
}

void Dropout2D::propagateLayer(int num) {
	for (int i = 0; i < numTokens[num]; i++) {
		for (int j = 0; j < size; j++) {
			float randValue = (float)rand() / (RAND_MAX + 1);
			if (randValue < dropoutRate) {
				neurons[num](i, j) = 0;
				dropped[num][i][j] = true;
			}
			else {
				neurons[num](i, j) = prevLayer->neurons[num](i, j) / dropoutRate;
				dropped[num][i][j] = false;
			}
		}
	}
}

void Dropout2D::backPropagate(int num) {
	for (int i = 0; i < numTokens[num]; i++) {
		for (int j = 0; j < size; j++) {
			if (!dropped[num][i][j]) {
				prevLayer->neuronGradient[num](i, j) = neuronGradient[num](i, j) / dropoutRate;
			}
			else {
				prevLayer->neuronGradient[num](i, j) = 0;
			}
		}
	}
	if (prevLayer != NULL) {
		prevLayer->backPropagate(num);
	}
}

void Dropout2D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	this->size = prevLayer->size;
	this->prevSize = size + 1;
}

void Dropout2D::setBatchSize(int batchSize) {
	Layer2D::initNeurons(batchSize);
	dropped = new bool** [batchSize];
	for (int i = 0; i < batchSize; i++) {
		dropped[i] = new bool* [maxNumTokens];
		for (int j = 0; j < maxNumTokens; j++) {
			dropped[i][j] = new bool[size];
		}
	}
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void Dropout2D::save(ofstream& file) {
	file << "Dropout," << dropoutRate << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void Dropout2D::predict(int num) {
	prevLayer->neurons[num].copy(batchSize, size, neurons[num]);
	if (nextLayer != NULL) {
		nextLayer->predict(num);
	}
}
