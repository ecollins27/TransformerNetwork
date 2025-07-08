#include "LayerNormalization2D.h"
#include "Model.h"
#include "ModelParser.h"
#include "MatrixKernel.h"

const string LayerNormalization2D::LAYER_NAME = "LayerNormalization2D";

__global__
void kernelMean(float* A, float* B, int batchNum, int height, int width) {
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
void kernelVariance(float* matrix, float* mean, float* output, int batchNum, int height, int width) {
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
void kernelNormalize(float* mean, float* std, float* A, float* B, int batchNum, int height, int N) {
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

void LayerNormalization2D::propagateLayer(int num) {
	MatrixKernel::runColumnKernel(numTokens[num], size, Matrix2::THREADS_PER_BLOCK * sizeof(float), kernelMean, prevLayer->neurons[num].device, mean.device, num, numTokens[num], size);
	MatrixKernel::runColumnKernel(numTokens[num], size, Matrix2::THREADS_PER_BLOCK * sizeof(float), kernelVariance, prevLayer->neurons[num].device, mean.device, variance.device, num, numTokens[num], size);
	variance.sqrt(std);
	MatrixKernel::runElementKernel(numTokens[num], size, 0, kernelNormalize, mean.device, std.device, prevLayer->neurons[num].device, neurons[num].device, num, numTokens[num], numTokens[num] * size);
	neurons[num].copyToHost();

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

__global__
void kernelBackPropagate(float c, float* mean, float* variance, float* std, float* neurons, float* neuronGradient, float* prevNeurons, float* prevNeuronGradient, int batchNum, int height, int width) {
	int i = blockIdx.x;
	int tid = threadIdx.x;
	int stride = blockDim.x;

	float sum;
	int id;
	for (int j = tid; j < width; j += stride) {
		id = i + height * j;
		if (std[j] != 0) {
			sum = 0;
			for (int k = 0; k < height; k++) {
				float grad = ((k == i ? 1 : 0) - c) - c * (prevNeurons[id] - mean[j]) * (prevNeurons[k + height * j] - mean[j]) / variance[j];
				grad /= std[j];
				sum += neuronGradient[k + height * j] * grad;
			}
			prevNeuronGradient[id] = sum;
		}
	}
}

void LayerNormalization2D::backPropagate(int num) {
	float c = 1.0 / numTokens[num];
	MatrixKernel::runRowKernel(batchSize, size, 0, kernelBackPropagate, c, mean.device, variance.device, std.device, neurons[num].device, neuronGradient[num].device, prevLayer->neurons[num].device, prevLayer->neuronGradient[num].device, num, numTokens[num], size);
	prevLayer->neuronGradient[num].copyToHost();
	//prevLayer->neuronGradient[num].constantFill(0, numTokens[num], size);
	//for (int i = 0; i < numTokens[num]; i++) {
	//	for (int j = 0; j < size; j++) {
	//		if (std(num, j) != 0) {
	//			for (int k = 0; k < numTokens[num]; k++) {
	//				float grad = ((k == i ? 1 : 0) - c) - c * (prevLayer->neurons[num](k, j) - mean(num, j)) * (prevLayer->neurons[num](i, j) - mean(num, j)) / variance(num, j);
	//				grad /= std(num, j);
	//				prevLayer->neuronGradient[num].r(i, j) += neuronGradient[num](k, j) * grad;
	//			}
	//		}
	//	}
	//}
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
	mean = Matrix2(batchSize, size, false);
	variance = Matrix2(batchSize, size, false);
	std = Matrix2(batchSize, size, false);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}