#include "Dropout1D.h"

Dropout1D::Dropout1D(float dropoutRate) {
	this->dropoutRate = dropoutRate;
}

void Dropout1D::propagateLayer(int num) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			float randValue = (float)rand() / (RAND_MAX + 1);
			if (randValue < dropoutRate) {
				neurons(i, j) = 0;
				dropped[i][j] = true;
			}
			else {
				neurons(i, j) = prevLayer->neurons(i, j) / dropoutRate;
				dropped[i][j] = false;
			}
		}
	}
}

void Dropout1D::backPropagate(int num) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			if (!dropped[i][j]) {
				prevLayer->neuronGradient(i, j) = neuronGradient(i, j) / dropoutRate;
			}
			else {
				prevLayer->neuronGradient(i, j) = 0;
			}
		}
	}
	if (prevLayer != NULL) {
		prevLayer->backPropagate(num);
	}
}

void Dropout1D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer1D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer1D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer1D*)prevLayer;
	this->size = prevLayer->size;
	this->prevSize = size + 1;
}

void Dropout1D::setBatchSize(int batchSize) {
	Layer1D::setBatchSize(batchSize);
	dropped = new bool* [batchSize];
	for (int i = 0; i < batchSize; i++) {
		dropped[i] = new bool[size];
		for (int j = 0; j < size; j++) {
			dropped[i][j] = false;
		}
	}
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void Dropout1D::save(ofstream& file){
	file << "Dropout," << dropoutRate << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void Dropout1D::predict(int num) {
	prevLayer->neurons.copy(batchSize, size, neurons);
	if (nextLayer != NULL) {
		nextLayer->predict(num);
	}
}