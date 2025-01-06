#include "PositionalEncoding2D.h"

PositionalEncoding2D::PositionalEncoding2D(float L) {
	this->L = L;
}

void PositionalEncoding2D::propagateLayer(int num) {
	for (int i = 0; i < numTokens[num]; i++) {
		for (int j = 0; j < size; j++) {
			if (j % 2 == 0) {
				neurons[num](i, j) = prevLayer->neurons[num](i, j) + sin(i / (pow(L, (float)j / size)));
			}
			else {
				neurons[num](i, j) = prevLayer->neurons[num](i, j) + cos(i / (pow(L, (float)(j - 1) / size)));
			}
		}
	}
}

void PositionalEncoding2D::backPropagate(int num) {
	neuronGradient[num].copy(numTokens[num], size, prevLayer->neuronGradient[num]);
	prevLayer->backPropagate(num);
}

void PositionalEncoding2D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	prevSize = prevLayer->size + 1;
}

void PositionalEncoding2D::save(ofstream& file) {
	file << "PositionalEncodingLayer," << L << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}