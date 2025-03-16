#include "Dropout1D.h"
#include "Model.h"
#include "ModelParser.h"

const string Dropout1D::LAYER_NAME = "Dropout1D";

Dropout1D::Dropout1D(float dropoutRate) {
	this->dropoutRate = dropoutRate;
	this->distribution = uniform_real_distribution<float>(0, 1);
}

Dropout1D::~Dropout1D() {
	for (int i = 0; i < batchSize; i++) {
		delete[] dropped[i];
	}
	delete[] dropped;
	Layer1D::~Layer1D();
}

void Dropout1D::propagateLayer(int num) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			float randValue = distribution(generator);
			if (randValue < dropoutRate) {
				neurons.r(i, j) = 0;
				dropped[i][j] = true;
			}
			else {
				neurons.r(i, j) = prevLayer->neurons(i, j) / dropoutRate;
				dropped[i][j] = false;
			}
		}
	}
}

void Dropout1D::backPropagate(int num) {
	if (num != 0) {
		prevLayer->backPropagate(num);
		return;
	}
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			if (!dropped[i][j]) {
				prevLayer->neuronGradient.r(i, j) = neuronGradient(i, j) / dropoutRate;
			}
			else {
				prevLayer->neuronGradient.r(i, j) = 0;
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
	file << LAYER_NAME << "," << dropoutRate << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void Dropout1D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Dropout1D* dropout = { new Dropout1D(ModelParser::getNextFloat(line, commaIndex, newCommaIndex)) };
	nn->addLayer(dropout);
}

void Dropout1D::predict(int num) {
	prevLayer->neurons.copy(batchSize, size, neurons);
	if (nextLayer != NULL) {
		nextLayer->predict(num);
	}
}