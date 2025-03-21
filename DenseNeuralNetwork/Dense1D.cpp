#include "Dense1D.h"
#include "Model.h"
#include "ModelParser.h"

const string Dense1D::LAYER_NAME = "Dense1D";

Dense1D::Dense1D(Activation* activation, int size) {
	this->activation = activation->clone();
	this->size = size;
}

Dense1D::~Dense1D() {
	delete activation;
	delete optimizer;
	weights.free();
	weightGradient.free();
	linearCombo.free();
	backPropIntermediate.free();
	Layer1D::~Layer1D();
}

void Dense1D::propagateLayer(int num) {
	Matrix::multiplyABtC(batchSize, prevSize, size, prevLayer->neurons, weights, linearCombo, true);
	activation->operate(batchSize, size, linearCombo, neurons);
}

void Dense1D::backPropagate(int num) {
	if (num != 0) {
		prevLayer->backPropagate(num);
		return;
	}
	activation->differentiate(batchSize, size, linearCombo, neurons, backPropIntermediate, neuronGradient);
	Matrix::multiplyABC(batchSize, size, prevSize, backPropIntermediate, weights, prevLayer->neuronGradient, true);
	Matrix::multiplyAtBC(size, batchSize, prevSize, backPropIntermediate, prevLayer->neurons, weightGradient, true);
	prevLayer->backPropagate(num);
}

void Dense1D::setPrevLayer(Layer* prevLayer) {
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
	weights = Matrix(new Matrix::NormalFill(0, stdDeviation), size, prevSize, true);
}

void Dense1D::setBatchSize(int batchSize) {
	Layer1D::setBatchSize(batchSize);
	linearCombo = Matrix(Matrix::ZERO_FILL, batchSize, size + 1, false);
	backPropIntermediate = Matrix(Matrix::ZERO_FILL, batchSize, size, true);
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
			denseLayer->weights.r(i, j) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
		}
	}
	*prevSize = size + 1;
}

void Dense1D::applyGradients(float learningRate, int t) {
	optimizer->applyGradient(weights, t, learningRate, batchSize);
	if (nextLayer != NULL) {
		nextLayer->applyGradients(learningRate, t);
	}
}

void Dense1D::setOptimizer(Optimizer* optimizer) {
	this->optimizer = optimizer->clone();
	this->optimizer->setDimensions(size, prevSize);
	weightGradient = this->optimizer->weightGradient;
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

int Dense1D::getNumParameters() {
	int current = nextLayer == NULL ? 0 : nextLayer->getNumParameters();
	return current + size * prevSize;
}