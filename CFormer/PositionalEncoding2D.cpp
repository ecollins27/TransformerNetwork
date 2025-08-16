// Compile with CUDA

#include "PositionalEncoding2D.h"
#include "Model.h"
#include "ModelParser.h"


const string PositionalEncoding2D::LAYER_NAME = "PositionalEncoding2D";

PositionalEncoding2D::PositionalEncoding2D(float L) {
	this->L = L;
}

void PositionalEncoding2D::initPropagationQueue(OperationQueue& queue) {
	for (int i = 0; i < batchSize; i++) {
		queue.enqueue(new PositionalEncodingOperation(L, prevLayer->neurons[i], neurons[i]));
	}
}

void PositionalEncoding2D::initBackPropQueue(OperationQueue& queue) {
	for (int i = 0; i < batchSize; i++) {
		queue.enqueue(new CopyTo(neuronGradient[i], prevLayer->neuronGradient[i]));
	}
	prevLayer->initBackPropQueue(queue);
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

__global__
void kernelEncoding(float L, float* A, float* B, int height, int width) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < width * height) {
		int row = i % height;
		int col = i / height;
		B[i] = A[i] + sin(row / (pow(L, (float)col / width)));
	}
}

bool PositionalEncodingOperation::operate(OperationQueue* queue, int threadID) {
	int N = this->A->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelEncoding << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (this->L, queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], this->A->height, this->A->width);
	return true;
}