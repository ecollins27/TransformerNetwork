// Compile with CUDA

#include "BatchNormalization1D.h"
#include "Model.h"
#include "ModelParser.h"


const string BatchNormalization1D::LAYER_NAME = "BatchNormalization1D";

BatchNormalization1D::BatchNormalization1D(float momentum) {
	this->momentum = momentum;
}

BatchNormalization1D::~BatchNormalization1D() {
	delete optimizer;
	Layer1D::~Layer1D();
}

void BatchNormalization1D::initPropagationQueue(OperationQueue& queue) {
	queue.enqueue(new BatchMeanOperation(prevLayer->neurons, batchMean));
	queue.enqueue(new BatchVarianceOperation(prevLayer->neurons, batchMean, batchVariance));
	queue.enqueue(new LinearCombo(momentum, mean, 1 - momentum, batchMean, mean));
	queue.enqueue(new LinearCombo(momentum, variance, 1 - momentum, batchVariance, variance));
	queue.enqueue(new Sqrt(variance, std));
	queue.enqueue(new BatchNormalizationOperation(prevLayer->neurons, mean, std, neurons));
}

void BatchNormalization1D::initBackPropQueue(OperationQueue& queue) {
	queue.enqueue(new ConstantFill(parameterGradient, 0));
	queue.enqueue(new BatchBackPropOperation((1 - momentum) / batchSize, prevLayer->neurons, neuronGradient, batchMean, mean, variance, std, parameters, prevLayer->neuronGradient, parameterGradient));
	prevLayer->initBackPropQueue(queue);
}

void BatchNormalization1D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer1D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer1D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer1D*)prevLayer;
	size = prevLayer->size;
	prevSize = size + 1;
	NormalFillFunction mean1Fill = NormalFillFunction(1, 1);
	parameters = Matrix(FillFunction::UNIT_NORMAL_FILL, 2, size);
	for (int i = 0; i < size; i++) {
		parameters(1, i) = mean1Fill(1, i);
	}
	mean = Matrix(1, size);
	batchMean = Matrix(1, size);
	variance = Matrix(1, size);
	batchVariance = Matrix(1, size);
	std = Matrix(1, size);
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
			batchNormalization->parameters(i, j) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
		}
	}
	ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
	for (int j = 0; j < *prevSize - 1; j++) {
		batchNormalization->mean(0, j) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
	}
	ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
	for (int j = 0; j < *prevSize - 1; j++) {
		batchNormalization->variance(0, j) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
	}
	for (int i = 0; i < batchNormalization->variance.height; i++) {
		for (int j = 0; j < batchNormalization->variance.width; j++) {
			batchNormalization->std(i, j) = batchNormalization->variance(i, j);
		}
	}
}

void BatchNormalization1D::initPredictQueue(OperationQueue& queue) {
	queue.enqueue(new BatchNormalizationOperation(prevLayer->neurons, mean, std, neurons));
	if (nextLayer != NULL) {
		nextLayer->initPredictQueue(queue);
	}
}

void BatchNormalization1D::initApplicationQueue(OperationQueue& queue, float learningRate, int& t) {
	this->optimizer->initApplicationQueue(queue, parameters, learningRate, batchSize, t);
	if (nextLayer != NULL) {
		nextLayer->initApplicationQueue(queue, learningRate, t);
	}
}

void BatchNormalization1D::setOptimizer(Optimizer<>* optimizer) {
	this->optimizer = optimizer->clone<Matrix>();
	this->optimizer->setDimensions(1, 2, size);
	parameterGradient = this->optimizer->weightGradient;
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

int BatchNormalization1D::getNumParameters() {
	return 2 * size;
}


__global__
void kernelBatchMean(float* A, float* B, int height, int width) {
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
		B[column] = shared[0] / height;
	}
}

bool BatchMeanOperation::operate(OperationQueue* queue, int threadID) {
	int N = this->A->length;
	int id = this->threadID.load();;
	kernelBatchMean << < this->A->width, Utils::THREADS_PER_BLOCK, Utils::THREADS_PER_BLOCK * sizeof(float), queue->streams[id] >> > (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], this->A->height, this->A->width);
	return true;
}

__global__
void kernelBatchVariance(float* matrix, float* mean, float* output, int height, int width) {
	extern __shared__ float shared[];

	int column = blockIdx.x;
	int tid = threadIdx.x;
	int stride = blockDim.x;

	float sum = 0.0f;
	for (int i = tid; i < height; i += stride) {
		sum += (matrix[i + height * column] - mean[column]) * (matrix[i + height * column] - mean[column]);
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
		output[column] = shared[0] / height;
	}
}

bool BatchVarianceOperation::operate(OperationQueue* queue, int threadID) {
	int N = this->A->length;
	int id = this->threadID.load();
	kernelBatchVariance << < this->A->width, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][2][0], this->A->height, this->A->width);
	return true;
}

__global__
void kernelParameterNormalize(float* A, float* mean, float* std, float* parameters, float* B, int height, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		int column = i / height;
		if (std[column] == 0) {
			B[i] = parameters[2 * column];
		}
		else {
			B[i] = parameters[2 * column] + parameters[2 * column + 1] * (A[i] - mean[column]) / std[column];
		}
	}
}

bool BatchNormalizationOperation::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelParameterNormalize << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][2][0], queue->hostDevices[id][3][0], queue->hostDevices[id][4][0], this->in[0]->height, N);
	return true;
}

__global__
void kernelBackPropagate(float c, float* prevNeurons, float* neuronGradient, float* batchMean, float* mean, float* variance, float* std, float* parameters, float* prevNeuronGradient, float* parameterGradient, int height, int width) {
	int i = blockIdx.x;
	int tid = threadIdx.x;
	int stride = blockDim.x;

	float sum;
	int id;
	for (int j = tid; j < width; j += stride) {
		id = i + height * j;
		parameterGradient[2 * j] += neuronGradient[id];
		if (std[j] != 0) {
			parameterGradient[2 * j + 1] += neuronGradient[id] * (prevNeurons[id] - mean[j]) / std[j];
			sum = 0;
			for (int k = 0; k < height; k++) {
				float grad = std[j] * ((k == i ? 1 : 0) - c) - (c / std[j]) * (prevNeurons[id] - batchMean[j]) * (prevNeurons[k + height * j] - mean[j]);
				sum += parameters[2 * j + 1] * neuronGradient[k + height * j] * grad / variance[j];
			}
			prevNeuronGradient[id] = sum;
		}
	}
}

bool BatchBackPropOperation::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cudaError_t err = cudaMemset(queue->hostDevices[id][8][0], 0, this->out[1]->length * sizeof(float));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory set failed: ") + cudaGetErrorString(err));
	}
	kernelBackPropagate <<< this->in[0]->height, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (this->c, queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][2][0], queue->hostDevices[id][3][0], queue->hostDevices[id][4][0], queue->hostDevices[id][5][0], queue->hostDevices[id][6][0], queue->hostDevices[id][7][0], queue->hostDevices[id][8][0], this->in[0]->height, this->in[0]->width);
	return true;
}