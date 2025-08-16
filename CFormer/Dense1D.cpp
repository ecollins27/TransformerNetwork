#include "Dense1D.h"
#include "Model.h"
#include "ModelParser.h"

const string Dense1D::LAYER_NAME = "Dense1D";

Dense1D::Dense1D(Activation* activation, int size) {
	this->activation = activation;
	this->size = size;
}

Dense1D::~Dense1D() {
	delete activation;
	delete optimizer;
	Layer1D::~Layer1D();
}

void Dense1D::initPropagationQueue(OperationQueue& queue) {
	queue.enqueue(new MultiplyABtC(prevLayer->neurons, weights, linearCombo, true));
	queue.enqueue(activation->getOperation(linearCombo, neurons));
}

void Dense1D::initBackPropQueue(OperationQueue& queue) {
	queue.enqueue(activation->getDifOperation(linearCombo, neurons, backPropIntermediate, neuronGradient));
	queue.enqueue(new MultiplyABC(backPropIntermediate, weights, prevLayer->neuronGradient, true));
	queue.enqueue(new MultiplyAtBC(backPropIntermediate, prevLayer->neurons, weightGradient, true));
	prevLayer->initBackPropQueue(queue);
}

void Dense1D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer1D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer1D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer1D*)prevLayer;
	prevSize = prevLayer->size + 1;
	float stdDeviation = sqrt(2.0 / (prevSize + size));
	if (activation->activationType == ActivationType::RELU || activation->activationType == ActivationType::ELU || activation->activationType == ActivationType::SWISH) {
		stdDeviation = sqrt(2.0 / prevSize);
	}
	else if (activation->activationType == ActivationType::SELU) {
		stdDeviation = sqrt(1.0 / prevSize);
	}
	NormalFillFunction fill = NormalFillFunction(0, stdDeviation);
	weights = Matrix(fill, size, prevSize);
}

void Dense1D::setBatchSize(int batchSize) {
	Layer1D::setBatchSize(batchSize);
	linearCombo = Matrix(batchSize, size);
	backPropIntermediate = Matrix(batchSize, size);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void Dense1D::save(ofstream& file) {
	file << LAYER_NAME << ",";
	activation->save(file);
	file << size << ",\n";
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < prevSize; j++) {
			file << weights(i, j) << ",";
		}
		file << "\n";
	}
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void Dense1D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Activation* activation = ModelParser::readActivation(line, commaIndex, newCommaIndex);
	int size = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	Dense1D* denseLayer = new Dense1D(activation, size);
	nn->addLayer(denseLayer);
	for (int i = 0; i < size; i++) {
		ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize; j++) {
			denseLayer->weights(i, j) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
		}
	}
	*prevSize = size + 1;
}

void Dense1D::initApplicationQueue(OperationQueue& queue, float learningRate, int& t) {
	this->optimizer->initApplicationQueue(queue, weights, learningRate, batchSize, t);
	if (nextLayer != NULL) {
		nextLayer->initApplicationQueue(queue, learningRate, t);
	}
}

void Dense1D::setOptimizer(Optimizer<>* optimizer) {
	this->optimizer = optimizer->clone<Matrix>();
	this->optimizer->setDimensions(1, size, prevSize);
	weightGradient = this->optimizer->weightGradient;
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

int Dense1D::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + size * prevSize;
}