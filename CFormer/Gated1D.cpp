#include "Gated1D.h"
#include "Model.h"
#include "ModelParser.h"

const string Gated1D::LAYER_NAME = "Gated1D";

Gated1D::Gated1D(Activation* activation, int size) {
	this->activation = activation->clone();
	this->size = size;
}

Gated1D::~Gated1D() {
	delete activation;
	delete optimizer1;
	delete optimizer2;
	Layer1D::~Layer1D();
}

void Gated1D::propagateLayer(int num) {
	Matrix2::multiplyABtC(prevLayer->neurons, weights1, A1, true);
	Matrix2::multiplyABtC(prevLayer->neurons, weights2, A2, true);
	activation->operate(A1, Ao);
	Matrix2::elementMultiply(Ao, A2, neurons);
}

void Gated1D::backPropagate(int num) {
	if (num != 0) {
		prevLayer->backPropagate(num);
		return;
	}
	Matrix2::elementMultiply(neuronGradient, A2, AoGrad);
	Matrix2::elementMultiply(neuronGradient, Ao, A2Grad);
	activation->differentiate(A1, Ao, A1Grad, AoGrad);
	Matrix2::multiplyABC(A1Grad, weights1, prevLayer->neuronGradient, true);
	Matrix2::multiplyABC(A2Grad, weights2, prevLayer->neuronGradient, false);
	Matrix2::multiplyAtBC(A1Grad, prevLayer->neurons, weightGradient1, true);
	Matrix2::multiplyAtBC(A2Grad, prevLayer->neurons, weightGradient2, true);
	prevLayer->backPropagate(num);
}

void Gated1D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer1D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer1D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer1D*)prevLayer;
	prevSize = prevLayer->size + 1;
	float stdDeviation = sqrt(2.0 / (prevSize + size));
	if (instanceOf<Relu>(activation) || instanceOf<Elu>(activation) || instanceOf<Swish>(activation)) {
		stdDeviation = sqrt(2.0 / prevSize);
	}
	else if (instanceOf<Selu>(activation)) {
		stdDeviation = sqrt(1.0 / prevSize);
	}
	NormalFill fill = NormalFill(0, stdDeviation);
	weights1 = Matrix2(fill, size, prevSize, 0);
	weights2 = Matrix2(fill, size, prevSize, 0);
}

void Gated1D::setBatchSize(int batchSize) {
	Layer1D::setBatchSize(batchSize);
	optimizer1->setBatchSize(batchSize, NULL);
	optimizer2->setBatchSize(batchSize, NULL);
	A1 = Matrix2(batchSize, size, 0);
	A1Grad = Matrix2(batchSize, size, 0);
	A2 = Matrix2(batchSize, size, 0);
	A2Grad = Matrix2(batchSize, size, 0);
	Ao = Matrix2(batchSize, size, 0);
	AoGrad = Matrix2(batchSize, size, 0);
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

void Gated1D::applyGradients(float learningRate, int t) {
	optimizer1->applyGradient(weights1, t, learningRate);
	optimizer2->applyGradient(weights2, t, learningRate);
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void Gated1D::setOptimizer(Optimizer* optimizer) {
	this->optimizer1 = optimizer->clone();
	this->optimizer1->setDimensions(size, prevSize);
	weightGradient1 = this->optimizer1->weightGradient;
	this->optimizer2 = optimizer->clone();
	this->optimizer2->setDimensions(size, prevSize);
	weightGradient2 = this->optimizer2->weightGradient;
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

int Gated1D::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + 2 * size * prevSize;
}