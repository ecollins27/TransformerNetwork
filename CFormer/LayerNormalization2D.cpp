#include "LayerNormalization2D.h"
#include "Model.h"
#include "ModelParser.h"
#include "MatrixKernel.h"

const string LayerNormalization2D::LAYER_NAME = "LayerNormalization2D";

__global__
void kernelMean(float* A, float* B, int height, int width) {
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
void kernelVariance(float* matrix, float* mean, float* output, int height, int width) {
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

__global__
void kernelNormalize(float* A, float* B, float* mean, float* std, int height, int N, int batchSize, int batchNum) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		int column = i / height;
		int bd = column * batchSize + batchNum;
		if (std[bd] == 0) {
			B[i] = 0;
		}
		else {
			B[i] = (A[i] - mean[bd]) / std[bd];
		}
	}
}

void LayerNormalization2D::propagateLayer(int num) {
	prevLayer->neurons[num].copyToDevice(0);
	MatrixKernel::runColumnKernel(numTokens[num], size, Utils::THREADS_PER_BLOCK * sizeof(float), kernelMean, Matrix2::DEVICES[num][0], Matrix2::DEVICES[0][1], numTokens[num], size);
	mean.copyToHost(1);
	MatrixKernel::runColumnKernel(numTokens[num], size, Utils::THREADS_PER_BLOCK * sizeof(float), kernelVariance, Matrix2::DEVICES[num][0], Matrix2::DEVICES[0][1], Matrix2::DEVICES[0][2], numTokens[num], size);
	variance.copyToHost(2);
	variance.sqrt(std);
	prevLayer->neurons[num].copyToDevice(0);
	mean.copyToDevice(2);
	std.copyToDevice(3);
	MatrixKernel::runElementKernel(numTokens[num], size, 0, kernelNormalize, Matrix2::DEVICES[num][0], Matrix2::DEVICES[num][1], Matrix2::DEVICES[0][2], Matrix2::DEVICES[0][3], numTokens[num], numTokens[num] * size, batchSize, num);
	neurons[num].copyToHost(1, numTokens[num] * size);
}

__global__
void kernelBackPropagate(float c, float* prevNeurons, float* prevNeuronGradient, float* neuronGradient, float* mean, float* variance, float* std, int height, int width, int batchSize, int batchNum) {
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

void LayerNormalization2D::backPropagate(int num) {
	float c = 1.0 / numTokens[num];
	prevLayer->neurons[num].copyToDevice(0);
	neuronGradient[num].copyToDevice(2);
	mean.copyToDevice(3);
	variance.copyToDevice(4);
	std.copyToDevice(5);
	MatrixKernel::runRowKernel(numTokens[num], size, 0, kernelBackPropagate, c, Matrix2::DEVICES[0][0], Matrix2::DEVICES[0][1], Matrix2::DEVICES[0][2], Matrix2::DEVICES[0][3], Matrix2::DEVICES[0][4], Matrix2::DEVICES[0][5], numTokens[num], size, batchSize, num);
	prevLayer->neuronGradient[num].copyToHost(1);
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
	mean = Matrix2(batchSize, size, 0);
	variance = Matrix2(batchSize, size, 0);
	std = Matrix2(batchSize, size, 0);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}