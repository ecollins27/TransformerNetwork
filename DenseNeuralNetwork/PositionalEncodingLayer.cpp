#include "PositionalEncodingLayer.h"

PositionalEncodingLayer::PositionalEncodingLayer() {
	L = 10000;
}

PositionalEncodingLayer::PositionalEncodingLayer(int L) {
	this->L = L;
}

void PositionalEncodingLayer::propagateLayer() {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			if (j % 2 == 0) {
				neurons[i][j] = prevLayer->neurons[i][j] + sin(i / (pow(L, (float)j / size)));
			}
			else {
				neurons[i][j] = prevLayer->neurons[i][j] + cos(i / (pow(L, (float)(j - 1) / size)));
			}
		}
	}
	Matrix::transpose(batchSize, size, neurons, neuronsTranspose);
}

void PositionalEncodingLayer::backPropagate() {
	Matrix::copy(batchSize, size, neuronGradient, prevLayer->neuronGradient);
	if (prevLayer != NULL) {
		prevLayer->backPropagate();
	}
}

void PositionalEncodingLayer::setPrevLayer(Layer* prevLayer) {
	this->prevLayer = prevLayer;
	size = prevLayer->size;
}

void PositionalEncodingLayer::save(ofstream& file) {
	file << "PositionalEncodingLayer," << L << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}