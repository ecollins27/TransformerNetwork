#include "Gated2D.h"
#include "Model.h"
#include "ModelParser.h"

const string Gated2D::LAYER_NAME = "Gated2D";

Gated2D::Gated2D(Activation* activation, int size) {
	this->activation = activation;
	this->size = size;
}

void Gated2D::initPropagationQueue(OperationQueue& queue) {
	for (int i = 0; i < batchSize; i++) {
		queue.enqueue(new MultiplyABtC(prevLayer->neurons[i], weights1, A1[i], true));
		queue.enqueue(new MultiplyABtC(prevLayer->neurons[i], weights2, A2[i], true));
		queue.enqueue(activation->getOperation(A1[i], Ao[i]));
		queue.enqueue(new ElementMultiply(Ao[i], A2[i], neurons[i]));
	}
}

void Gated2D::initBackPropQueue(OperationQueue& queue) {
	queue.enqueue(new ConstantFill(weightGradient1, 0));
	queue.enqueue(new ConstantFill(weightGradient2, 0));
	for (int i = 0; i < batchSize; i++) {
		queue.enqueue(new ElementMultiply(neuronGradient[i], A2[i], AoGrad[i]));
		queue.enqueue(new ElementMultiply(neuronGradient[i], Ao[i], A2Grad[i]));
		queue.enqueue(activation->getDifOperation(A1[i], Ao[i], A1Grad[i], AoGrad[i]));
		queue.enqueue(new MultiplyABC(A1Grad[i], weights1, prevLayer->neuronGradient[i], true));
		queue.enqueue(new MultiplyAtBC(A1Grad[i], prevLayer->neurons[i], weightGradient1, false));
		queue.enqueue(new MultiplyABC(A2Grad[i], weights2, prevLayer->neuronGradient[i], false));
		queue.enqueue(new MultiplyAtBC(A2Grad[i], prevLayer->neurons[i], weightGradient2, false));
	}
	prevLayer->initBackPropQueue(queue);
}

void Gated2D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
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

void Gated2D::setBatchSize(int batchSize) {
	Layer2D::initNeurons(batchSize);
	weightGradient1 = Matrix(size, prevSize);
	weightGradient2 = Matrix(size, prevSize);

	A1 = Matrix::allocateMatrixArray(batchSize, maxNumTokens, size);
	A1Grad = Matrix::allocateMatrixArray(batchSize, maxNumTokens, size);
	A2 = Matrix::allocateMatrixArray(batchSize, maxNumTokens, size);
	A2Grad = Matrix::allocateMatrixArray(batchSize, maxNumTokens, size);
	Ao = Matrix::allocateMatrixArray(batchSize, maxNumTokens, size);
	AoGrad = Matrix::allocateMatrixArray(batchSize, maxNumTokens, size);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void Gated2D::save(ofstream& file) {
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

void Gated2D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Activation* activation = ModelParser::readActivation(line, commaIndex, newCommaIndex);
	int size = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	Gated2D* gatedLayer = { new Gated2D(activation, size) };
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

void Gated2D::setNumTokens(int* numTokens) {
	this->numTokens = numTokens;
	updateNeuronDimensions();
	int height;
	for (int i = 0; i < batchSize; i++) {
		height = numTokens[i];
		A1[i].setHeight(height);
		A1Grad[i].setHeight(height);
		A2[i].setHeight(height);
		A2Grad[i].setHeight(height);
		Ao[i].setHeight(height);
		AoGrad[i].setHeight(height);
	}
	if (nextLayer != NULL && instanceOf<Layer2D>(nextLayer)) {
		((Layer2D*)nextLayer)->setNumTokens(numTokens);
	}
}

void Gated2D::initApplicationQueue(OperationQueue& queue, float learningRate, int& t) {
	optimizer1->initApplicationQueue(queue, weights1, learningRate, batchSize, t);
	optimizer2->initApplicationQueue(queue, weights2, learningRate, batchSize, t);
	if (nextLayer != NULL) {
		nextLayer->initApplicationQueue(queue, learningRate, t);
	}
}

void Gated2D::setOptimizer(Optimizer<>* optimizer) {
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

int Gated2D::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + 2 * size * prevSize;
}