#include "Dropout2D.h"
#include "Model.h"
#include "ModelParser.h"

const string Dropout2D::LAYER_NAME = "Dropout2D";

Dropout2D::Dropout2D(float dropoutRate) {
	this->dropoutRate = dropoutRate;
	this->distribution = uniform_real_distribution<float>(0, 1);
}

void Dropout2D::propagateLayer(int num) {
	for (int i = 0; i < numTokens[num]; i++) {
		for (int j = 0; j < size; j++) {
			float randValue = distribution(generator);
			if (randValue < dropoutRate) {
				neurons[num].r(i, j) = 0;
				dropped[num][i][j] = true;
			}
			else {
				neurons[num].r(i, j) = prevLayer->neurons[num](i, j) / dropoutRate;
				dropped[num][i][j] = false;
			}
		}
	}
}

void Dropout2D::backPropagate(int num) {
	for (int i = 0; i < numTokens[num]; i++) {
		for (int j = 0; j < size; j++) {
			if (!dropped[num][i][j]) {
				prevLayer->neuronGradient[num].r(i, j) = neuronGradient[num](i, j) / dropoutRate;
			}
			else {
				prevLayer->neuronGradient[num].r(i, j) = 0;
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
	file << LAYER_NAME << "," << dropoutRate << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void Dropout2D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Dropout2D* dropout = { new Dropout2D(ModelParser::getNextFloat(line, commaIndex, newCommaIndex)) };
	nn->addLayer(dropout);
}

void Dropout2D::predict(int num) {
	prevLayer->neurons[num].copy(numTokens[num], size, neurons[num]);
	if (nextLayer != NULL) {
		nextLayer->predict(num);
	}
}
