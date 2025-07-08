#include "SequenceMean.h"
#include "Model.h"
#include "ModelParser.h"
#include "MatrixKernel.h"

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

__global__
void kernelColumnMean(float* A, float* B, int height, int width, int batch) {
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
		B[batch + height * column] = shared[0] / height;
	}
}

void SequenceMean::propagateLayer(int num) {
	MatrixKernel::runColumnKernel(prevLayer->numTokens[num], size, Matrix2::THREADS_PER_BLOCK * sizeof(float), kernelColumnMean, prevLayer->neurons[num].device, neurons.device, prevLayer->numTokens[num], size, num);
	forwardThreadCount.fetch_add(1);
	if (forwardThreadCount.load() >= batchSize && num == 0) {
		neurons.copyToHost();
		forwardThreadCount.store(0);
		activation->operate(means, neurons);
		gradientCalculated.store(false);
		if (nextLayer != NULL) {
			nextLayer->forwardPropagate(num);
		}
	}
}

__global__
void kernelBackPropagate(float c, float* prevNeuronGradient, float* backPropIntermediate, int size, int N, int batchNum) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		int column = i % size;
		prevNeuronGradient[i] = c * backPropIntermediate[batchNum * size + column];
	}
}

void SequenceMean::backPropagate(int num) {
	if (num == 0) {
		activation->differentiate(means, neurons, backPropIntermediate, neuronGradient);
		gradientCalculated.store(true);
	}
	while (!gradientCalculated.load()){}
	float c = 1.0 / prevLayer->numTokens[num];
	MatrixKernel::runElementKernel(prevLayer->numTokens[num], size, 0, kernelBackPropagate, c, prevLayer->neuronGradient[num].device, backPropIntermediate.device, size, prevLayer->numTokens[num] * size, num);
	prevLayer->neuronGradient[num].copyToHost();
	//for (int i = 0; i < prevLayer->numTokens[num]; i++) {
	//	for (int j = 0; j < size; j++) {
	//		prevLayer->neuronGradient[num].r(i, j) = c * backPropIntermediate(num, j);
	//	}
	//}
	prevLayer->backPropagate(num);
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
	means = Matrix2(batchSize, size, false);
	backPropIntermediate = Matrix2(batchSize, size, true);
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