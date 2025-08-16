// Compile with CUDA

#include "LayerNormalization2D.h"
#include "Model.h"
#include "ModelParser.h"


const string LayerNormalization2D::LAYER_NAME = "LayerNormalization2D";

void LayerNormalization2D::initPropagationQueue(OperationQueue& queue) {
	for (int i = 0; i < batchSize; i++) {
		queue.enqueue(new LayerNormalizationOperation(prevLayer->neurons[i], means[i], variances[i], stds[i], neurons[i]));
	}
}

__global__
void kernelBackPropagate(float c, float* prevNeurons, float* neuronGradient, float* mean, float* variance, float* std, float* prevNeuronGradient, int height, int width, int batchSize, int batchNum) {
	int i = blockIdx.x;
	int tid = threadIdx.x;
	int stride = blockDim.x;

	float sum;
	int id, bid;
	for (int j = tid; j < width; j += stride) {
		id = i + height * j;
		bid = j * batchSize + batchNum;
		if (std[bid] != 0) {
			sum = 0;
			for (int k = 0; k < height; k++) {
				float grad = std[bid] * ((k == i ? 1 : 0) - c) - (c / std[bid]) * (prevNeurons[id] - mean[bid]) * (prevNeurons[k + height * j] - mean[bid]);
				sum += neuronGradient[k + height * j] * grad / variance[bid];
			}
			prevNeuronGradient[id] = sum;
		}
	}
}

void LayerNormalization2D::initBackPropQueue(OperationQueue& queue) {
	for (int i = 0; i < batchSize; i++) {
		queue.enqueue(new LayerNormalizationBackPropOperation(prevLayer->neurons[i], neuronGradient[i], means[i], variances[i], stds[i], prevLayer->neuronGradient[i]));
	}
	prevLayer->initBackPropQueue(queue);
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
	means = Matrix::allocateMatrixArray(batchSize, 1, size + 1);
	variances = Matrix::allocateMatrixArray(batchSize, 1, size + 1);
	stds = Matrix::allocateMatrixArray(batchSize, 1, size + 1);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

__global__
void kernelLayerMean(float* A, float* B, int height, int width) {
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

__global__
void kernelLayerVariance(float* matrix, float* mean, float* output, float* outputSqrt, int height, int width) {
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
		outputSqrt[column] = sqrt(output[column]);
	}
}

__global__
void kernelLayerNormalize(float* A, float* mean, float* std, float* B, int height, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		int column = i / height;
		if (std[column] == 0) {
			B[i] = 0;
		}
		else {
			B[i] = (A[i] - mean[column]) / std[column];
		}
	}
}

// Matrix& input, Matrix& mean, Matrix& variance, Matrix& std, Matrix& output
bool LayerNormalizationOperation::operate(OperationQueue* queue, int threadID) {
	int height = this->in[0]->height;
	int width = this->in[0]->width;
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelLayerMean << < numBlocks, Utils::THREADS_PER_BLOCK, Utils::THREADS_PER_BLOCK * sizeof(float), queue->streams[id] >> > (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], height, width);
	kernelLayerVariance << < numBlocks, Utils::THREADS_PER_BLOCK, Utils::THREADS_PER_BLOCK * sizeof(float), queue->streams[id] >> > (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][2][0], queue->hostDevices[id][3][0], height, width);
	kernelLayerNormalize << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][2][0], queue->hostDevices[id][4][0], height, width);
	return true;
}

__global__
void kernelLayerBackPropagate(float c, float* prevNeurons, float* neuronGradient, float* mean, float* variance, float* std, float* prevNeuronGradient, int height, int width) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < height * width) {
		float c = 1.0 / height;
		int j = id / height;
		float sum, grad;
		if (std[j] != 0) {
			sum = 0;
			for (int k = 0; k < height; k++) {
				grad = std[j] * ((k == blockIdx.x ? 1 : 0) - c) - (c / std[j]) * (prevNeurons[id] - mean[j]) * (prevNeurons[k + height * j] - mean[j]);
				sum += neuronGradient[k + height * j] * grad / variance[j];
			}
			prevNeuronGradient[id] = sum;
		}
		else {
			prevNeuronGradient[id] = 0;
		}
	}
}

// Matrix& input, Matrix& outputGrad, Matrix& mean, Matrix& variance, Matrix& std, Matrix& inputGrad
bool LayerNormalizationBackPropOperation::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length - this->in[0]->height;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelLayerBackPropagate << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (1.0 / this->in[0]->height, queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][2][0], queue->hostDevices[id][3][0], queue->hostDevices[id][4][0], queue->hostDevices[id][5][0], this->in[0]->height, this->in[0]->width - 1);
	return true;
}