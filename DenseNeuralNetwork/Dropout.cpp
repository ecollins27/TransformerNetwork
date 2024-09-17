#include "Dropout.h"

Dropout::Dropout(double dropRate) {
	this->dropRate = dropRate;
	this->scale = 1.0 / (1 - dropRate);
}

void Dropout::setPrevLayer(Layer* layer) {
	this->prevLayer = layer;
	this->size = prevLayer->size;
}

void Dropout::setNextLayer(Layer* layer) {
	this->nextLayer = layer;
}

void Dropout::setBatchSize(int batchSize) {
	this->batchSize = batchSize;
	neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
	for (int i = 0; i < batchSize; i++) {
		neurons[i][size] = 1;
	}
	neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
	dropped = (bool**)malloc(batchSize * sizeof(bool*));
	for (int i = 0; i < batchSize; i++) {
		dropped[i] = (bool*)malloc(size * sizeof(bool));
	}
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void Dropout::predict() {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			neurons[i][j] = prevLayer->neurons[i][j];
		}
	}
	if (nextLayer != NULL) {
		nextLayer->predict();
	}
}

void Dropout::forwardPropagate() {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			double randValue = (double)rand() / (RAND_MAX + 1);
			if (randValue < dropRate) {
				neurons[i][j] = 0;
				dropped[i][j] = true;
			}
			else {
				neurons[i][j] = prevLayer->neurons[i][j];
				dropped[i][j] = false;
			}
		}
	}
	if (nextLayer != NULL) {
		nextLayer->forwardPropagate();
	}
}

void Dropout::backPropagate() {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			if (!dropped[i][j]) {
				prevLayer->neuronGradient[i][j] = neuronGradient[i][j] / scale;
			}
		}
	}
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void Dropout::applyGradients(TrainingParams* params, int t) {
	if (nextLayer != NULL) {
		nextLayer->applyGradients(params, t);
	}
}

void Dropout::save(ofstream& file) {
	file << "Dropout," << dropRate << "\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}