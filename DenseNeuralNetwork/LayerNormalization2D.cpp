#include "LayerNormalization2D.h"
#include "Model.h"
#include "ModelParser.h"

const string LayerNormalization2D::LAYER_NAME = "LayerNormalization2D";

void LayerNormalization2D::propagateLayer(int num) {
	Matrix::calculateMean(numTokens[num], size, prevLayer->neurons[num], mean, num);
	Matrix::calculateVariance(numTokens[num], size, prevLayer->neurons[num], mean, variance, num);
	variance.sqrt(batchSize, size, std, num);
	Matrix::normalize(numTokens[num], size, prevLayer->neurons[num], neurons[num], mean, std, num);

	//for (int j = 0; j < size; j++) {
	//	float& meanSum = mean.r(num, j);
	//	meanSum = 0;
	//	for (int i = 0; i < numTokens[num]; i++) {
	//		meanSum += prevLayer->neurons[num](i, j);
	//	}
	//	meanSum /= numTokens[num];

	//	float& varianceSum = variance.r(num, j);
	//	varianceSum = 0;
	//	for (int i = 0; i < numTokens[num]; i++) {
	//		varianceSum += (prevLayer->neurons[num](i, j) - meanSum) * (prevLayer->neurons[num](i, j) - meanSum);
	//	}
	//	varianceSum /= numTokens[num];
	//	std.r(num, j) = sqrt(varianceSum);

	//	for (int i = 0; i < numTokens[num]; i++) {
	//		if (std(num, j) == 0) {
	//			neurons[num].r(i, j) = 0;
	//		}
	//		else {
	//			neurons[num].r(i, j) = (prevLayer->neurons[num](i, j) - mean(num, j)) / std(num, j);
	//		}
	//	}
	//}
}

void LayerNormalization2D::backPropagate(int num) {
	float c = 1.0 / numTokens[num];
	prevLayer->neuronGradient[num].constantFill(0, numTokens[num], size);
	for (int i = 0; i < numTokens[num]; i++) {
		for (int j = 0; j < size; j++) {
			if (std(num, j) != 0) {
				for (int k = 0; k < numTokens[num]; k++) {
					float grad = ((k == i ? 1 : 0) - c) - c * (prevLayer->neurons[num](k, j) - mean(num, j)) * (prevLayer->neurons[num](i, j) - mean(num, j)) / variance(num, j);
					grad /= std(num, j);
					prevLayer->neuronGradient[num].r(i, j) += neuronGradient[num](k, j) * grad;
				}
			}
		}
	}
	prevLayer->backPropagate(num);
}

void LayerNormalization2D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	size = prevLayer->size;
	prevSize = size + 1;
}

void LayerNormalization2D::save(ofstream& file) {
	file << LAYER_NAME << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void LayerNormalization2D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	LayerNormalization2D* layerNormalization = { new LayerNormalization2D() };
	nn->addLayer(layerNormalization);
}

void LayerNormalization2D::setBatchSize(int batchSize) {
	Layer2D::initNeurons(batchSize);
	mean = Matrix(Matrix::ZERO_FILL, batchSize, size, false);
	variance = Matrix(Matrix::ZERO_FILL, batchSize, size, false);
	std = Matrix(Matrix::ZERO_FILL, batchSize, size, false);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}