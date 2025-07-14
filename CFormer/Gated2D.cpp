#include "Gated2D.h"
#include "Model.h"
#include "ModelParser.h"

const string Gated2D::LAYER_NAME = "Gated2D";

Gated2D::Gated2D(Activation* activation, int size) {
	this->activation = activation->clone();
	this->size = size;
}

void Gated2D::propagateLayer(int num) {
	Matrix2::multiplyABtC(prevLayer->neurons[num], weights1, A1[num], true);
	Matrix2::multiplyABtC(prevLayer->neurons[num], weights2, A2[num], true);
	activation->operate(A1[num], Ao[num]);
	Matrix2::elementMultiply(Ao[num], A2[num], neurons[num]);
}

void Gated2D::backPropagate(int num) {
	Matrix2::elementMultiply(neuronGradient[num], A2[num], AoGrad[num]);
	Matrix2::elementMultiply(neuronGradient[num], Ao[num], A2Grad[num]);
	activation->differentiate(A1[num], Ao[num], A1Grad[num], AoGrad[num]);
	Matrix2::multiplyABC(A1Grad[num], weights1, prevLayer->neuronGradient[num], true);
	Matrix2::multiplyABC(A2Grad[num], weights2, prevLayer->neuronGradient[num], false);
	Matrix2::multiplyAtBC(A1Grad[num], prevLayer->neurons[num], weightGradient1[num], true);
	Matrix2::multiplyAtBC(A2Grad[num], prevLayer->neurons[num], weightGradient2[num], true);
	prevLayer->backPropagate(num);
}

void Gated2D::setPrevLayer(Layer* prevLayer) {
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
	weights1 = Matrix2(fill, size, prevSize);
	weights2 = Matrix2(fill, size, prevSize);
}

void Gated2D::setBatchSize(int batchSize) {
	Layer2D::initNeurons(batchSize);
	weightGradient1 = Matrix2::allocateMatrixArray(batchSize, size, prevSize, false);
	weightGradient2 = Matrix2::allocateMatrixArray(batchSize, size, prevSize, false);

	optimizer1->setBatchSize(batchSize, weightGradient1);
	optimizer2->setBatchSize(batchSize, weightGradient2);

	A1 = Matrix2::allocateMatrixArray(batchSize, maxNumTokens, size, false);
	A1Grad = Matrix2::allocateMatrixArray(batchSize, maxNumTokens, size, false);
	A2 = Matrix2::allocateMatrixArray(batchSize, maxNumTokens, size, false);
	A2Grad = Matrix2::allocateMatrixArray(batchSize, maxNumTokens, size, false);
	Ao = Matrix2::allocateMatrixArray(batchSize, maxNumTokens, size, false);
	AoGrad = Matrix2::allocateMatrixArray(batchSize, maxNumTokens, size, false);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void Gated2D::save(ofstream& file) {
	file << LAYER_NAME << ",";
	activation->save(file);
	file << size << ",\n";
	weights1.allocateHost();
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < prevSize; j++) {
			file << weights1(i, j) << ",";
		}
		file << "\n";
	}
	weights1.deallocateHost();
	weights2.allocateHost();
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < prevSize; j++) {
			file << weights2(i, j) << ",";
		}
		file << "\n";
	}
	weights2.deallocateHost();
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void Gated2D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Activation* activation = ModelParser::readActivation(line, commaIndex, newCommaIndex);
	int size = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	Gated2D* gatedLayer = { new Gated2D(activation, size) };
	nn->addLayer(gatedLayer);
	gatedLayer->weights1.allocateHost();
	for (int i = 0; i < size; i++) {
		ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize; j++) {
			gatedLayer->weights1(i, j) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
		}
	}
	gatedLayer->weights1.deallocateHost();
	gatedLayer->weights2.allocateHost();
	for (int i = 0; i < size; i++) {
		ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize; j++) {
			gatedLayer->weights2(i, j) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
		}
	}
	gatedLayer->weights2.deallocateHost();
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

void Gated2D::applyGradients(float learningRate, int t) {
	optimizer1->condenseGradients();
	optimizer2->condenseGradients();
	optimizer1->applyGradient(weights1, t, learningRate);
	optimizer2->applyGradient(weights2, t, learningRate);
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void Gated2D::setOptimizer(Optimizer* optimizer) {
	this->optimizer1 = optimizer->clone();
	this->optimizer1->setDimensions(size, prevSize);
	this->optimizer2 = optimizer->clone();
	this->optimizer2->setDimensions(size, prevSize);
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

int Gated2D::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + 2 * size * prevSize;
}