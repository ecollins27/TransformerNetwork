#include "PositionalEncoding2D.h"
#include "Model.h"
#include "ModelParser.h"
#include "MatrixKernel.h"

const string PositionalEncoding2D::LAYER_NAME = "PositionalEncoding2D";

PositionalEncoding2D::PositionalEncoding2D(float L) {
	this->L = L;
}

__global__
void kernelEncoding(float L, float* A, float* B, int height, int width) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < width * height) {
		int row = i % height;
		int col = i / height;
		B[i] = A[i] + sin(row / (pow(L, (float)col / width)));
	}
}

void PositionalEncoding2D::propagateLayer(int num) {
	MatrixKernel::runElementKernel(numTokens[num], size, 0, kernelEncoding, L, prevLayer->neurons[num].device, neurons[num].device, numTokens[num], size);
	neurons[num].copyToHost();
	//for (int i = 0; i < numTokens[num]; i++) {
	//	for (int j = 0; j < size; j++) {
	//		if (j % 2 == 0) {
	//			neurons[num].r(i, j) = prevLayer->neurons[num](i, j) + sin(i / (pow(L, (float)j / size)));
	//		}
	//		else {
	//			neurons[num].r(i, j) = prevLayer->neurons[num](i, j) + cos(i / (pow(L, (float)(j - 1) / size)));
	//		}
	//	}
	//}
}

void PositionalEncoding2D::backPropagate(int num) {
	prevLayer->neuronGradient[num].copy(neuronGradient[num]);
	prevLayer->backPropagate(num);
}

void PositionalEncoding2D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	prevSize = prevLayer->size + 1;
	size = prevLayer->size;
}

void PositionalEncoding2D::save(ofstream& file) {
	file << LAYER_NAME << "," << L << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void PositionalEncoding2D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	int L = ModelParser::getNextInt(line, commaIndex, newCommaIndex);
	PositionalEncoding2D* positionalEncodingLayer = { new PositionalEncoding2D(L) };
	nn->addLayer(positionalEncodingLayer);
}