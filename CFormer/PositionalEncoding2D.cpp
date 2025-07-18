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
	prevLayer->neurons[num].copyToDevice(0);
	MatrixKernel::runElementKernel(numTokens[num], size, 0, kernelEncoding, L, Matrix2::DEVICES[num][0], Matrix2::DEVICES[num][1], numTokens[num], size);
	neurons[num].copyToHost(1);
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