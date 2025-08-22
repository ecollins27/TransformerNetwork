#include "Gated1D.h"
#include "Model.h"
#include "ModelParser.h"

const string Gated1D::LAYER_NAME = "Gated1D";

Gated1D::Gated1D(Activation* activation, int size) {
	this->activation = activation;
	this->size = size;
}

Gated1D::~Gated1D() {
	delete activation;
	delete optimizer1;
	delete optimizer2;
	Layer1D::~Layer1D();
}

void Gated1D::initPropagationQueue(OperationQueue& queue) {
	queue.enqueue(new MultiplyABtC(prevLayer->neurons, weights1, A1, true));
	queue.enqueue(new MultiplyABtC(prevLayer->neurons, weights2, A2, true));
	queue.enqueue(activation->getOperation(A1, Ao));
	queue.enqueue(new ElementMultiply(Ao, A2, neurons));
}

void Gated1D::initBackPropQueue(OperationQueue& queue) {
	queue.enqueue(new ElementMultiply(neuronGradient, A2, AoGrad));
	queue.enqueue(new ElementMultiply(neuronGradient, Ao, A2Grad));
	queue.enqueue(activation->getDifOperation(A1, Ao, A1Grad, A2Grad));
	queue.enqueue(new MultiplyABC(A1Grad, weights1, prevLayer->neuronGradient, true));
	queue.enqueue(new MultiplyAtBC(A1Grad, prevLayer->neurons, weightGradient1, true));
	queue.enqueue(new MultiplyABC(A2Grad, weights2, prevLayer->neuronGradient, false));
	queue.enqueue(new MultiplyAtBC(A2Grad, prevLayer->neurons, weightGradient2, true));
	prevLayer->initBackPropQueue(queue);
}

void Gated1D::setPrevLayer(Layer* prevLayer) {
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
	weights1 = Matrix(fill, size, prevSize);
	weights2 = Matrix(fill, size, prevSize);
}

void Gated1D::setBatchSize(int batchSize) {
	Layer1D::setBatchSize(batchSize);
	A1 = Matrix(batchSize, size);
	A1Grad = Matrix(batchSize, size);
	A2 = Matrix(batchSize, size);
	A2Grad = Matrix(batchSize, size);
	Ao = Matrix(batchSize, size);
	AoGrad = Matrix(batchSize, size);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void Gated1D::save(ofstream& file) {
	file << LAYER_NAME << ",";
	activation->save(file);
	file << size << ",\n";
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < prevSize; j++) {
			file << weights1(i, j) << ",";
		}
		file << "\n";
	}
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < prevSize; j++) {
			file << weights2(i, j) << ",";
		}
		file << "\n";
	}
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void Gated1D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Activation* activation = ModelParser::readActivation(line, commaIndex, newCommaIndex);
	int size = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	Gated1D* gatedLayer = { new Gated1D(activation, size) };
	nn->addLayer(gatedLayer);
	for (int i = 0; i < size; i++) {
		ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize; j++) {
			gatedLayer->weights1(i, j) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
		}
	}
	for (int i = 0; i < size; i++) {
		ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize; j++) {
			gatedLayer->weights2(i, j) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
		}
	}
	*prevSize = size + 1;
}

void Gated1D::initApplicationQueue(OperationQueue& queue, float learningRate, int& t){
	this->optimizer1->initApplicationQueue(queue, weights1, learningRate, batchSize, t);
	this->optimizer2->initApplicationQueue(queue, weights2, learningRate, batchSize, t);
	if (nextLayer != NULL) {
		nextLayer->initApplicationQueue(queue, learningRate, t);
	}
}

void Gated1D::setOptimizer(Optimizer<>* optimizer) {
	this->optimizer1 = (Optimizer<Matrix>*) optimizer->clone(true);
	this->optimizer1->setDimensions(1, size, prevSize);
	weightGradient1 = this->optimizer1->weightGradient;
	this->optimizer2 = (Optimizer<Matrix>*) optimizer->clone(true);
	this->optimizer2->setDimensions(1, size, prevSize);
	weightGradient2 = this->optimizer2->weightGradient;
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

int Gated1D::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + 2 * size * prevSize;
}