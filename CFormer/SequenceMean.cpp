// Compile with CUDA

#include "SequenceMean.h"
#include "Model.h"
#include "ModelParser.h"


const string SequenceMean::LAYER_NAME = "SequenceMean";

SequenceMean::SequenceMean(Activation* activation) {
	this->activation = activation;
	forwardThreadCount = { 0 };
	backThreadCount = { 0 };
	gradientCalculated = { false };
}

SequenceMean::~SequenceMean() {
	delete activation;
	Layer1D::~Layer1D();
}

void SequenceMean::initPropagationQueue(OperationQueue& queue) {
	queue.enqueue(new MeanCondenseOperation(batchSize, prevLayer->neurons, neurons));
}

void SequenceMean::initBackPropQueue(OperationQueue& queue) {
	queue.enqueue(new MeanCondenseBackPropOperation(batchSize, neuronGradient, prevLayer->neuronGradient));
}

void SequenceMean::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer2D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer2D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer2D*)prevLayer;
	size = prevLayer->size;
	prevSize = size + 1;
}

void SequenceMean::setBatchSize(int batchSize) {
	Layer1D::setBatchSize(batchSize);
	means = Matrix(batchSize, size + 1);
	backPropIntermediate = Matrix(batchSize, size + 1);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void SequenceMean::save(ofstream& file) {
	file << LAYER_NAME << ",";
	activation->save(file);
	file << ",\n";
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void SequenceMean::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Activation* activation = ModelParser::readActivation(line, commaIndex, newCommaIndex);
	SequenceMean* batchSum = { new SequenceMean(activation) };
	nn->addLayer(batchSum);
}

__global__
void kernelSequenceMean(float* A, float* B, int batchSize, int height, int width, int row) {
	extern __shared__ float shared[];

	int column = blockIdx.x;
	int tid = threadIdx.x;
	int stride = blockDim.x;

	float sum = 0.0f;
	for (int i = tid; i < height; i += stride) {
		sum += A[i + height * column];
	}

	shared[tid] = sum;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shared[tid] += shared[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		B[column * batchSize + row] = shared[0] / height;
	}
}

bool MeanCondenseOperation::operate(OperationQueue* queue, int threadID) {
	int N;
	int id = this->threadID.load();
	int width = this->in[0]->width;
	for (int i = 0; i < N_IN; i++) {
		N = this->in[i]->length;
		kernelSequenceMean << < width, Utils::THREADS_PER_BLOCK, Utils::THREADS_PER_BLOCK * sizeof(float), queue->streams[id] >> > (queue->hostDevices[id][i][0], queue->hostDevices[id][N_IN][0], N_IN, this->in[i]->height, width, i);
	}
	return true;
}

__global__
void kernelSequenceBackPropagate(float* backPropIntermediate, float* prevNeuronGradient, int batchSize, int height, int N, int row) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		int column = i / height;
		prevNeuronGradient[i] = backPropIntermediate[row + batchSize * column] / height;
	}
}

bool MeanCondenseBackPropOperation::operate(OperationQueue* queue, int threadID) {
	int N;
	int id = this->threadID.load();
	int numBlocks;
	for (int i = 0; i < N_OUT; i++) {
		N = this->out[i]->length;
		numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
		kernelSequenceBackPropagate << < N, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->hostDevices[id][0][0], queue->hostDevices[id][i + 1][0], N_OUT, this->out[i]->height, N, i);
	}
	return true;
}