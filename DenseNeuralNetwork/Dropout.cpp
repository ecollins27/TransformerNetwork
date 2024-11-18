#include "Dropout.h"

Dropout::Dropout(float dropRate) {
	this->dropRate = dropRate;
	this->scale = 1.0 / (1 - dropRate);
}

Dropout::~Dropout() {
	Matrix::deallocateMatrix(neurons, batchSize, size + 1);
	Matrix::deallocateMatrix(neuronsTranspose, size + 1, batchSize);
	Matrix::deallocateMatrix(neuronGradient, batchSize, size + 1);
	for (int i = 0; i < batchSize; i++) {
		free(dropped[i]);
	}
	free(dropped);
}

void Dropout::setPrevLayer(Layer* layer) {
	this->prevLayer = layer;
	this->size = prevLayer->size;
}

void Dropout::setBatchSize(int batchSize) {
	if (maxBatchSize > 0) {
		this->batchSize = batchSize;
	}
	else {
		if (neurons != NULL) {
			Matrix::deallocateMatrix(neurons, batchSize, size + 1);
			Matrix::deallocateMatrix(neuronsTranspose, size + 1, batchSize);
			Matrix::deallocateMatrix(neuronGradient, batchSize, size + 1);
			for (int i = 0; i < batchSize; i++) {
				free(dropped[i]);
			}
			free(dropped);
		}
		this->batchSize = batchSize;
		neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
		neuronsTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, batchSize);
		for (int i = 0; i < batchSize; i++) {
			neurons[i][size] = 1;
		}
		neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, batchSize, size + 1);
		dropped = (bool**)malloc(batchSize * sizeof(bool*));
		for (int i = 0; i < batchSize; i++) {
			dropped[i] = (bool*)malloc(size * sizeof(bool));
		}
	}
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void Dropout::setMaxBatchSize(int maxBatchSize) {
	if (maxBatchSize > 0) {
		if (neurons != NULL) {
			Matrix::deallocateMatrix(neurons, maxBatchSize, size + 1);
			Matrix::deallocateMatrix(neuronsTranspose, size + 1, maxBatchSize);
			Matrix::deallocateMatrix(neuronGradient, maxBatchSize, size + 1);
			for (int i = 0; i < maxBatchSize; i++) {
				free(dropped[i]);
			}
			free(dropped);
		}
		this->maxBatchSize = maxBatchSize;
		neurons = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size + 1);
		neuronsTranspose = Matrix::allocateMatrix(Matrix::ZERO_FILL, size + 1, maxBatchSize);
		for (int i = 0; i < maxBatchSize; i++) {
			neurons[i][size] = 1;
		}
		neuronGradient = Matrix::allocateMatrix(Matrix::ZERO_FILL, maxBatchSize, size + 1);
		dropped = (bool**)malloc(maxBatchSize * sizeof(bool*));
		for (int i = 0; i < maxBatchSize; i++) {
			dropped[i] = (bool*)malloc(size * sizeof(bool));
		}
	} else {
		this->maxBatchSize = maxBatchSize;
	}
	if (nextLayer != NULL) {
		nextLayer->setMaxBatchSize(maxBatchSize);
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
			float randValue = (float)rand() / (RAND_MAX + 1);
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
	Matrix::transpose(batchSize, size + 1, neurons, neuronsTranspose);
	if (nextLayer != NULL) {
		nextLayer->forwardPropagate();
	}
}

void Dropout::backPropagate() {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			if (!dropped[i][j]) {
				prevLayer->neuronGradient[i][j] = neuronGradient[i][j];
			} else {
				prevLayer->neuronGradient[i][j] = 0;
			}
		}
	}
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void Dropout::save(ofstream& file) {
	file << "Dropout," << dropRate << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}