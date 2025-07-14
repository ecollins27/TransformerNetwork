#include "Dense2D.h"
#include "Model.h"
#include "ModelParser.h"

const string Dense2D::LAYER_NAME = "Dense2D";

Dense2D::Dense2D(Activation* activation, int size) {
	this->activation = activation->clone();
	this->size = size;
}

void Dense2D::propagateLayer(int num) {
	Matrix2::multiplyABtC(prevLayer->neurons[num], weights, linearCombo[num], true);
	activation->operate(linearCombo[num], neurons[num]);
}

void Dense2D::backPropagate(int num) {
	activation->differentiate(linearCombo[num], neurons[num], backPropIntermediate[num], neuronGradient[num]);
	Matrix2::multiplyABC(backPropIntermediate[num], weights, prevLayer->neuronGradient[num], true);
	Matrix2::multiplyAtBC(backPropIntermediate[num], prevLayer->neurons[num], weightGradient[num], true);
	prevLayer->backPropagate(num);
}

void Dense2D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	prevSize = prevLayer->size + 1;
	float stdDeviation = sqrt(2.0 / (prevSize + size));
	if (instanceOf<Relu>(activation) || instanceOf<Elu>(activation) || instanceOf<Swish>(activation)) {
		stdDeviation = sqrt(2.0 / prevSize);
	}
	else if (instanceOf<Selu>(activation)) {
		stdDeviation = sqrt(1.0 / prevSize);
	}
	FillFunction fill = NormalFill(0, stdDeviation);
	weights = Matrix2(fill, size, prevSize);
}

void Dense2D::setBatchSize(int batchSize) {
	Layer2D::initNeurons(batchSize);
	weightGradient = Matrix2::allocateMatrixArray(batchSize, size, prevSize, false);
	linearCombo = Matrix2::allocateMatrixArray(batchSize, maxNumTokens, size, false);
	backPropIntermediate = Matrix2::allocateMatrixArray(batchSize, maxNumTokens, size, false);
	optimizer->setBatchSize(batchSize, weightGradient);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void Dense2D::save(ofstream& file) {
	file << LAYER_NAME << ",";
	activation->save(file);
	file << size << ",\n";
	weights.allocateHost();
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < prevSize; j++) {
			file << weights(i, j) << ",";
		}
		file << "\n";
	}
	weights.deallocateHost();
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void Dense2D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Activation* activation = ModelParser::readActivation(line, commaIndex, newCommaIndex);
	int size = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	Dense2D* denseLayer = new Dense2D(activation, size);
	nn->addLayer(denseLayer);
	denseLayer->weights.allocateHost();
	for (int i = 0; i < size; i++) {
		ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize; j++) {
			denseLayer->weights(i, j) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
		}
	}
	denseLayer->weights.deallocateHost();
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

void Dense2D::applyGradients(float learningRate, int t) {
	optimizer->condenseGradients();
	optimizer->applyGradient(weights, t, learningRate);
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void Dense2D::setOptimizer(Optimizer* optimizer) {
	this->optimizer = optimizer->clone();
	this->optimizer->setDimensions(size, prevSize);
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

int Dense2D::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + size * prevSize;
}