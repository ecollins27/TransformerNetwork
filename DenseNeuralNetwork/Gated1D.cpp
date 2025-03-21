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
	weights1.free();
	weightGradient1.free();
	weights2.free();
	weightGradient2.free();
	A1.free();
	A1Grad.free();
	A2.free();
	A2Grad.free();
	Ao.free();
	AoGrad.free();
	Layer1D::~Layer1D();
}

void Gated1D::propagateLayer(int num) {
	Matrix::multiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights1, A1, true);
	Matrix::multiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights2, A2, true);
	activation->operate(batchSize, size, A1, Ao);
	Matrix::elementMultiply(batchSize, size, Ao, A2, neurons);
}

void Gated1D::backPropagate(int num) {
	if (num != 0) {
		prevLayer->backPropagate(num);
		return;
	}
	Matrix::elementMultiply(batchSize, size, neuronGradient, A2, AoGrad);
	Matrix::elementMultiply(batchSize, size, neuronGradient, Ao, A2Grad);
	activation->differentiate(batchSize, size, A1, Ao, A1Grad, AoGrad);
	Matrix::multiplyABC(batchSize, size, prevSize, A1Grad, weights1, prevLayer->neuronGradient, true);
	Matrix::multiplyABC(batchSize, size, prevSize, A2Grad, weights2, prevLayer->neuronGradient, false);
	Matrix::multiplyAtBC(size, batchSize, prevSize, A1Grad, prevLayer->neurons, weightGradient1, true);
	Matrix::multiplyAtBC(size, batchSize, prevSize, A2Grad, prevLayer->neurons, weightGradient2, true);
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
	weights1 = Matrix(new Matrix::NormalFill(0, stdDeviation), size, prevSize, true);
	weights2 = Matrix(new Matrix::NormalFill(0, stdDeviation), size, prevSize, true);
}

void Gated1D::setBatchSize(int batchSize) {
	Layer1D::setBatchSize(batchSize);
	A1 = Matrix(Matrix::ZERO_FILL, batchSize, size, false);
	A1Grad = Matrix(Matrix::ZERO_FILL, batchSize, size, true);
	A2 = Matrix(Matrix::ZERO_FILL, batchSize, size, false);
	A2Grad = Matrix(Matrix::ZERO_FILL, batchSize, size, true);
	Ao = Matrix(Matrix::ZERO_FILL, batchSize, size, false);
	AoGrad = Matrix(Matrix::ZERO_FILL, batchSize, size, false);
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
			gatedLayer->weights1.r(i, j) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
		}
	}
	for (int i = 0; i < size; i++) {
		ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize; j++) {
			gatedLayer->weights2.r(i, j) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
		}
	}
	*prevSize = size + 1;
}

void Gated1D::applyGradients(float learningRate, int t) {
	optimizer1->applyGradient(weights1, t, learningRate, batchSize);
	optimizer2->applyGradient(weights2, t, learningRate, batchSize);
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