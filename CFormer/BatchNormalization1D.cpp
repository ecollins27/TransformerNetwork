#include "BatchNormalization1D.h"
#include "Model.h"
#include "ModelParser.h"
#include "MatrixKernel.h"

const string BatchNormalization1D::LAYER_NAME = "BatchNormalization1D";

BatchNormalization1D::BatchNormalization1D(float momentum) {
	this->momentum = momentum;
}

BatchNormalization1D::~BatchNormalization1D() {
	delete optimizer;
	Layer1D::~Layer1D();
}

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
void kernelParameterNormalize(float* mean, float* std, float* parameters, float* A, float* B, int height, int N) {
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

void BatchNormalization1D::propagateLayer(int num) {
	MatrixKernel::runColumnKernel(batchSize, size, Matrix2::THREADS_PER_BLOCK * sizeof(float), kernelMean, prevLayer->neurons.device, batchMean.device, batchSize, size);
	MatrixKernel::runColumnKernel(batchSize, size, Matrix2::THREADS_PER_BLOCK * sizeof(float), kernelVariance, prevLayer->neurons.device, batchMean.device, batchVariance.device, batchSize, size);
	Matrix2::linearCombo(momentum, mean, 1 - momentum, batchMean, mean);
	Matrix2::linearCombo(momentum, variance, 1 - momentum, batchVariance, variance);
	variance.sqrt(std);
	MatrixKernel::runElementKernel(batchSize, size, 0, kernelParameterNormalize, mean.device, std.device, parameters.device, prevLayer->neurons.device, neurons.device, batchSize, batchSize * size);
	neurons.copyToHost();

	//for (int j = 0; j < size; j++) {
	//	float& meanSum = batchMean.r(0, j);
	//	meanSum = 0;
	//	for (int i = 0; i < batchSize; i++) {
	//		meanSum += prevLayer->neurons(i, j);
	//	}
	//	meanSum /= batchSize;
	//	mean.r(0, j) = momentum * mean(0, j) + (1 - momentum) * meanSum;

	//	float& varianceSum = batchVariance.r(0, j);
	//	varianceSum = 0;
	//	for (int i = 0; i < batchSize; i++) {
	//		varianceSum += (prevLayer->neurons(i, j) - meanSum) * (prevLayer->neurons(i, j) - meanSum);
	//	}
	//	varianceSum /= batchSize;
	//	variance.r(0, j) = momentum * variance(0, j) + (1 - momentum) * varianceSum;
	//	std.r(0, j) = sqrt(variance(0, j));

	//	for (int i = 0; i < batchSize; i++) {
	//		if (std(0, j) == 0) {
	//			neurons.r(i, j) = parameters(0, j);
	//		}
	//		else {
	//			neurons.r(i, j) = parameters(0, j) + parameters(1, j) * (prevLayer->neurons(i, j) - mean(0, j)) / std(0, j);
	//		}
	//	}
	//}
}

__global__
void kernelBackPropagate(float c, float* batchMean, float* mean, float* variance, float* std, float* parameters, float* parameterGradient, float* neurons, float* neuronGradient, float* prevNeurons, float* prevNeuronGradient, int height, int width) {
	int i = blockIdx.x;
	int tid = threadIdx.x;
	int stride = blockDim.x;

	float sum;
	int id;
	for (int j = tid; j < width; j += stride) {
		id = i + height * j;
		parameterGradient[2 * j] += neuronGradient[id];
		if (std[j] != 0) {
			parameterGradient[2 * j + 1] += neuronGradient[id] * neurons[id];
			sum = 0;
			for (int k = 0; k < height; k++) {
				float grad = std[j] * ((k == i ? 1 : 0) - c) - (c / std[j]) * (prevNeurons[id] - batchMean[j]) * (prevNeurons[k + height * j] - mean[j]);
				sum += parameters[2 * j + 1] * neuronGradient[k + height * j] * grad / variance[j];
			}
			prevNeuronGradient[id] = sum;
		}
	}
}

void BatchNormalization1D::backPropagate(int num) {
	if (num != 0) {
		prevLayer->backPropagate(num);
		return;
	}
	parameterGradient.constantFill(0);
	float c = (1 - momentum) / batchSize;
	MatrixKernel::runRowKernel(batchSize, size, 0, kernelBackPropagate, c, batchMean.device, mean.device, variance.device, std.device, parameters.device, parameterGradient.device, neurons.device, neuronGradient.device, prevLayer->neurons.device, prevLayer->neuronGradient.device, batchSize, size);
	prevLayer->neuronGradient.copyToHost();
	//prevLayer->neuronGradient.constantFill(0);
	//for (int i = 0; i < batchSize; i++) {
	//	for (int j = 0; j < size; j++) {
	//		parameterGradient.r(0, j) += neuronGradient(i, j);
	//		if (std(0, j) != 0) {
	//			parameterGradient.r(1, j) += neuronGradient(i,j) * (prevLayer->neurons(i, j) - mean(0, j)) / std(0, j);
	//			for (int k = 0; k < batchSize; k++) {
	//				float grad = ((k == i ? 1 : 0) - c) - c * (prevLayer->neurons(i, j) - batchMean(0, j)) * (prevLayer->neurons(k, j) - mean(0, j)) / variance(0, j);
	//				grad /= std(0, j);
	//				prevLayer->neuronGradient.r(i, j) += parameters(1, j) * neuronGradient(k, j) * grad;
	//			}
	//		}
	//	}
	//}
	prevLayer->backPropagate(num);
}

void BatchNormalization1D::setPrevLayer(Layer* prevLayer) {
	if (!instanceOf<Layer1D>(prevLayer)) {
		throw invalid_argument("Previous layer must be instance Layer1D");
	}
	index = prevLayer->index + 1;
	this->prevLayer = (Layer1D*)prevLayer;
	size = prevLayer->size;
	prevSize = size + 1;
	NormalFill mean1Fill = NormalFill(1, 1);
	parameters = Matrix2(FillFunction::UNIT_NORMAL_FILL, 2, size);
	for (int i = 0; i < size; i++) {
		parameters(1, i) = mean1Fill(1, i);
	}
	parameters.deallocateHost();
	mean = Matrix2(1, size, false);
	batchMean = Matrix2(1, size, false);
	variance = Matrix2(1, size, false);
	batchVariance = Matrix2(1, size, false);
	std = Matrix2(1, size, false);
}

void BatchNormalization1D::setBatchSize(int batchSize) {
	Layer1D::setBatchSize(batchSize);
	optimizer->setBatchSize(batchSize, NULL);
	if (nextLayer != NULL) {
		nextLayer->setBatchSize(batchSize);
	}
}

void BatchNormalization1D::save(ofstream& file) {
	file << LAYER_NAME.c_str() << ",\n";
	parameters.allocateHost();
	mean.allocateHost();
	variance.allocateHost();
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
	parameters.deallocateHost();
	mean.deallocateHost();
	variance.deallocateHost();
	if (nextLayer != NULL) {
		nextLayer->save(file);
	}
}

void BatchNormalization1D::load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	BatchNormalization1D* batchNormalization = { new BatchNormalization1D(0.9) };
	nn->addLayer(batchNormalization);
	batchNormalization->parameters.allocateHost();
	for (int i = 0; i < 2; i++) {
		ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize - 1; j++) {
			batchNormalization->parameters(i, j) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
		}
	}
	batchNormalization->parameters.deallocateHost();
	batchNormalization->mean.allocateHost();
	ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
	for (int j = 0; j < *prevSize - 1; j++) {
		batchNormalization->mean(0, j) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
	}
	batchNormalization->mean.deallocateHost();
	batchNormalization->variance.allocateHost();
	ModelParser::getNextLine(file, line, commaIndex, newCommaIndex);
	for (int j = 0; j < *prevSize - 1; j++) {
		batchNormalization->variance(0, j) = ModelParser::getNextfloat(line, commaIndex, newCommaIndex);
	}
	batchNormalization->variance.deallocateHost();
	batchNormalization->variance.sqrt(batchNormalization->std);
}

void BatchNormalization1D::predict(int num) {
	MatrixKernel::runElementKernel(batchSize, size, 0, kernelParameterNormalize, mean.device, std.device, parameters.device, prevLayer->neurons.device, neurons.device, batchSize, batchSize * size);
	neurons.copyToHost();
	//for (int j = 0; j < size; j++) {
	//	for (int i = 0; i < batchSize; i++) {
	//		if (std(0, j) == 0) {
	//			neurons.r(i, j) = parameters(0, j);
	//		}
	//		else {
	//			neurons.r(i, j) = parameters(0, j) + parameters(1, j) * (prevLayer->neurons(i, j) - mean(0, j)) / std(0, j);
	//		}
	//	}
	//}
	if (nextLayer != NULL) {
		nextLayer->predict(num);
	}
}

void BatchNormalization1D::applyGradients(float learningRate, int t) {
	this->optimizer->applyGradient(parameters, t, learningRate);
}

void BatchNormalization1D::setOptimizer(Optimizer* optimizer) {
	this->optimizer = optimizer->clone();
	this->optimizer->setDimensions(2, size);
	parameterGradient = this->optimizer->weightGradient;
	if (nextLayer != NULL) {
		nextLayer->setOptimizer(optimizer);
	}
}

int BatchNormalization1D::getNumParameters() {
	return 2 * size;
}