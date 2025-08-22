#include "Dense2D.h"
#include "Model.h"
#include "ModelParser.h"

const string Dense2D::LAYER_NAME = "Dense2D";

Dense2D::Dense2D(Activation* activation, int size) {
	this->activation = activation;
	this->size = size;
}

void Dense2D::initPropagationQueue(OperationQueue& queue) {
	for (int i = 0; i < batchSize; i++) {
		queue.enqueue(new MultiplyABtC(prevLayer->neurons[i], weights, linearCombo[i], true));
		queue.enqueue(activation->getOperation(linearCombo[i], neurons[i]));
	}
}

void Dense2D::initBackPropQueue(OperationQueue& queue) {
	queue.enqueue(new ConstantFill(weightGradient, 0));
	for (int i = 0; i < batchSize; i++) {
		queue.enqueue(activation->getDifOperation(linearCombo[i], neurons[i], backPropIntermediate[i], neuronGradient[i]));
		queue.enqueue(new MultiplyABC(backPropIntermediate[i], weights, prevLayer->neuronGradient[i], true));
		queue.enqueue(new MultiplyAtBC(backPropIntermediate[i], prevLayer->neurons[i], weightGradient, false));
	}
	prevLayer->initBackPropQueue(queue);
}

void Dense2D::setPrevLayer(Layer* prevLayer) {
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
	weights = Matrix(fill, size, prevSize);
}

void Dense2D::setBatchSize(int batchSize) {
	Layer2D::initNeurons(batchSize);
	linearCombo = Matrix::allocateMatrixArray(batchSize, maxNumTokens, size);
	backPropIntermediate = Matrix::allocateMatrixArray(batchSize, maxNumTokens, size);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void Dense2D::save(ofstream& file) {
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

void Dense2D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Activation* activation = ModelParser::readActivation(line, commaIndex, newCommaIndex);
	int size = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	Dense2D* denseLayer = new Dense2D(activation, size);
	nn->addLayer(denseLayer);
	for (int i = 0; i < size; i++) {
		ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize; j++) {
			denseLayer->weights(i, j) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
		}
	}
	*prevSize = size + 1;
}

void Dense2D::setNumTokens(int* numTokens) {
	this->numTokens = numTokens;
	updateNeuronDimensions();
	for (int i = 0; i < batchSize; i++) {
		linearCombo[i].setHeight(numTokens[i]);
		backPropIntermediate[i].setHeight(numTokens[i]);
	}
	if (nextLayer != NULL && instanceOf<Layer2D>(nextLayer)) {
		((Layer2D*)nextLayer)->setNumTokens(numTokens);
	}
}

void Dense2D::initApplicationQueue(OperationQueue& queue, float learningRate, int& t) {
	optimizer->initApplicationQueue(queue, weights, learningRate, batchSize, t);
	if (nextLayer != NULL) {
		nextLayer->initApplicationQueue(queue, learningRate, t);
	}
}

void Dense2D::setOptimizer(Optimizer<>* optimizer) {
	this->optimizer = (Optimizer<Matrix>*) optimizer->clone(true);
	this->optimizer->setDimensions(1, size, prevSize);
	weightGradient = this->optimizer->weightGradient;
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

int Dense2D::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + size * prevSize;
}