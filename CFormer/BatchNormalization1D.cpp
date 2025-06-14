#include "BatchNormalization1D.h"
#include "Model.h"
#include "ModelParser.h"

const string BatchNormalization1D::LAYER_NAME = "BatchNormalization1D";

BatchNormalization1D::BatchNormalization1D(float momentum) {
	this->momentum = momentum;
}

BatchNormalization1D::~BatchNormalization1D() {
	delete optimizer;
	mean.free();
	batchMean.free();
	variance.free();
	batchVariance.free();
	std.free();
	parameters.free();
	parameterGradient.free();
	Layer1D::~Layer1D();
}

void BatchNormalization1D::propagateLayer(int num) {
	Matrix::calculateMean(batchSize, size, prevLayer->neurons, batchMean, 0);
	Matrix::calculateVariance(batchSize, size, prevLayer->neurons, batchMean, batchVariance, 0);
	Matrix::linearCombo(1, size, momentum, mean, 1 - momentum, batchMean, mean);
	Matrix::linearCombo(1, size, momentum, variance, 1 - momentum, batchVariance, variance);
	variance.sqrt(1, size, std, 0);
	Matrix::parameterNormalize(batchSize, size, prevLayer->neurons, neurons, mean, std, parameters, 0);

	//for (int j = 0; j < size; j++) {
	//	float& meanSum = batchMean.r(0, j);
	//	meanSum = 0;
	//	for (int i = 0; i < batchSize; i++) {
	//		meanSum += prevLayer->neurons(i, j);
	//	}
	//	meanSum /= batchSize;
	//	mean.r(0, j) = momentum * mean(0, j) + (1 - momentum) * meanSum;

	//	float& varianceSum = batchVariance.r(0, j);
	//	varianceSum = 0;
	//	for (int i = 0; i < batchSize; i++) {
	//		varianceSum += (prevLayer->neurons(i, j) - meanSum) * (prevLayer->neurons(i, j) - meanSum);
	//	}
	//	varianceSum /= batchSize;
	//	variance.r(0, j) = momentum * variance(0, j) + (1 - momentum) * varianceSum;
	//	std.r(0, j) = sqrt(variance(0, j));

	//	for (int i = 0; i < batchSize; i++) {
	//		if (std(0, j) == 0) {
	//			neurons.r(i, j) = parameters(0, j);
	//		}
	//		else {
	//			neurons.r(i, j) = parameters(0, j) + parameters(1, j) * (prevLayer->neurons(i, j) - mean(0, j)) / std(0, j);
	//		}
	//	}
	//}
}

void BatchNormalization1D::backPropagate(int num) {
	if (num != 0) {
		prevLayer->backPropagate(num);
		return;
	}
	float c = (1 - momentum) / batchSize;
	prevLayer->neuronGradient.constantFill(0, batchSize, size);
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < size; j++) {
			parameterGradient.r(0, j) += neuronGradient(i, j);
			if (std(0, j) != 0) {
				parameterGradient.r(1, j) += neuronGradient(i,j) * (prevLayer->neurons(i, j) - mean(0, j)) / std(0, j);
				for (int k = 0; k < batchSize; k++) {
					float grad = ((k == i ? 1 : 0) - c) - c * (prevLayer->neurons(i, j) - batchMean(0, j)) * (prevLayer->neurons(k, j) - mean(0, j)) / variance(0, j);
					grad /= std(0, j);
					prevLayer->neuronGradient.r(i, j) += parameters(1, j) * neuronGradient(k, j) * grad;
				}
			}
		}
	}
	prevLayer->backPropagate(num);
}

void BatchNormalization1D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer1D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer1D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer1D*)prevLayer;
	size = prevLayer->size;
	prevSize = size + 1;
	parameters = Matrix(Matrix::UNIT_NORMAL_FILL, 2, size, false);
	mean = Matrix(Matrix::ZERO_FILL, 1, size, false);
	batchMean = Matrix(Matrix::ZERO_FILL, 1, size, false);
	variance = Matrix(Matrix::ZERO_FILL, 1, size, false);
	batchVariance = Matrix(Matrix::ZERO_FILL, 1, size, false);
	std = Matrix(Matrix::ZERO_FILL, 1, size, false);
}

void BatchNormalization1D::setBatchSize(int batchSize) {
	Layer1D::setBatchSize(batchSize);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void BatchNormalization1D::save(ofstream& file) {
	file << LAYER_NAME.c_str() << ",\n";
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < size; j++) {
			if (i < 2) {
				file << parameters(i, j) << ",";
			}
			else if (i == 2) {
				file << mean(0, j) << ",";
			}
			else {
				file << variance(0, j) << ",";
			}
		}
		file << "\n";
	}
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void BatchNormalization1D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	BatchNormalization1D* batchNormalization = { new BatchNormalization1D(0.9) };
	nn->addLayer(batchNormalization);
	for (int i = 0; i < 2; i++) {
		ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize - 1; j++) {
			batchNormalization->parameters.r(i, j) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
		}
	}
	ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
	for (int j = 0; j < *prevSize - 1; j++) {
		batchNormalization->mean.r(0, j) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
	}
	ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
	for (int j = 0; j < *prevSize - 1; j++) {
		batchNormalization->variance.r(0, j) = ModelParser::getNextFloat(line, commaIndex, newCommaIndex);
		batchNormalization->std.r(0, j) = sqrt(batchNormalization->variance(0, j) + 0.0000001);
	}
}

void BatchNormalization1D::predict(int num) {
	for (int j = 0; j < size; j++) {
		for (int i = 0; i < batchSize; i++) {
			if (std(0, j) == 0) {
				neurons.r(i, j) = parameters(0, j);
			}
			else {
				neurons.r(i, j) = parameters(0, j) + parameters(1, j) * (prevLayer->neurons(i, j) - mean(0, j)) / std(0, j);
			}
		}
	}
	if (nextLayer != NULL) {
		nextLayer->predict(num);
	}
}

void BatchNormalization1D::applyGradients(float learningRate, int t) {
	this->optimizer->applyGradient(parameters, t, learningRate, batchSize);
}

void BatchNormalization1D::setOptimizer(Optimizer* optimizer) {
	this->optimizer = optimizer->clone();
	this->optimizer->setDimensions(2, size);
	parameterGradient = this->optimizer->weightGradient;
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

int BatchNormalization1D::getNumParameters() {
	return 2 * size;
}