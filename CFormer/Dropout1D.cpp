#include "Dropout1D.h"
#include "Model.h"
#include "ModelParser.h"

const string Dropout1D::LAYER_NAME = "Dropout1D";

Dropout1D::Dropout1D(float dropoutRate) {
	this->dropoutRate = dropoutRate;
}

Dropout1D::~Dropout1D() {
	for (int i = 0; i < batchSize; i++) {
		delete[] dropped[i];
	}
	delete[] dropped;
	Layer1D::~Layer1D();
}

void Dropout1D::initPropagationQueue(OperationQueue& queue) {
	queue.enqueue(new DropoutForwardPropOperation(dropoutRate, prevLayer->neurons, neurons, dropped));
}

void Dropout1D::initBackPropQueue(OperationQueue& queue) {
	queue.enqueue(new DropoutBackPropOperation(dropoutRate, neuronGradient, prevLayer->neuronGradient, dropped));
	prevLayer->initBackPropQueue(queue);
}

void Dropout1D::initPredictQueue(OperationQueue& queue) {
	queue.enqueue(new CopyTo(prevLayer->neurons, neurons));
	if (nextLayer != NULL) {
		nextLayer->initPredictQueue(queue);
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
	Dropout1D* dropout = { new Dropout1D(ModelParser::getNextfloat(line, commaIndex, newCommaIndex)) };
	nn->addLayer(dropout);
}

template<>
bool DropoutForwardPropOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	for (int i = 0; i < this->A->height; i++) {
		for (int j = 0; j < this->A->width; j++) {
			float randValue = distribution(generator);
			if (randValue < dropoutRate) {
				this->B->operator()(i, j) = 0;
				dropped[i][j] = true;
			}
			else {
				this->B->operator()(i, j) = this->A->operator()(i, j) / dropoutRate;
				dropped[i][j] = false;
			}
		}
		if (this->B->isLayerOutput) {
			this->B->operator()(i, this->A->width - 1) = 1;
		}
	}
	return true;
}

template<>
bool DropoutForwardPropOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	for (int i = 0; i < this->A->batchSize; i++) {
		for (int j = 0; j < this->A->length; j++) {
			if (this->B->isLayerOutput && j / this->A->height == this->A->width - 1) {
				this->B->host[i][j] = 1;
			}
			else {
				float randValue = distribution(generator);
				if (randValue < dropoutRate) {
					this->B->host[i][j] = 0;
					dropped[i][j] = true;
				}
				else {
					this->B->host[i][j] = this->A->host[i][j] / dropoutRate;
					dropped[i][j] = false;
				}
			}
		}
	}
	return true;
}

template<>
bool DropoutBackPropOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	for (int i = 0; i < this->A->height; i++) {
		for (int j = 0; j < this->A->width; j++) {
			if (!dropped[i][j]) {
				this->B->operator()(i, j) = this->A->operator()(i, j) / dropoutRate;
			}
			else {
				this->B->operator()(i, j) = 0;
			}
		}
	}
	return true;
}

template<>
bool DropoutBackPropOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	for (int i = 0; i < this->A->batchSize; i++) {
		for (int j = 0; j < this->A->length; j++) {
			if (!dropped[i][j]) {
				this->B->host[i][j] = this->A->host[i][j] / dropoutRate;
			}
			else {
				this->B->host[i][j] = 0;
			}
		}
	}
	return true;
}