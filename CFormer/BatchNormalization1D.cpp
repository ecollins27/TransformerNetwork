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

__global__
void kernelParameterNormalize(float* A, float* B, float* mean, float* std, float* parameters, int height, int N) {
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
	prevLayer->neurons.copyToDevice(0);
	MatrixKernel::runColumnKernel(batchSize, size, Utils::THREADS_PER_BLOCK * sizeof(float), kernelBatchMean, Matrix2::DEVICES[0][0], Matrix2::DEVICES[0][1], batchSize, size);
	batchMean.copyToHost(1);
	MatrixKernel::runColumnKernel(batchSize, size, Utils::THREADS_PER_BLOCK * sizeof(float), kernelBatchVariance, Matrix2::DEVICES[0][0], Matrix2::DEVICES[0][1], Matrix2::DEVICES[0][2], batchSize, size);
	batchVariance.copyToHost(2);
	Matrix2::linearCombo(momentum, mean, 1 - momentum, batchMean, mean);
	Matrix2::linearCombo(momentum, variance, 1 - momentum, batchVariance, variance);
	variance.sqrt(std);
	prevLayer->neurons.copyToDevice(0);
	mean.copyToDevice(2);
	std.copyToDevice(3);
	parameters.copyToDevice(4);
	MatrixKernel::runElementKernel(batchSize, size, 0, kernelParameterNormalize, Matrix2::DEVICES[0][0], Matrix2::DEVICES[0][1], Matrix2::DEVICES[0][2], Matrix2::DEVICES[0][3], Matrix2::DEVICES[0][4], batchSize, batchSize * size);
	neurons.copyToHost(1, batchSize * size);
}

__global__
void kernelBackPropagate(float c, float* prevNeurons, float* prevNeuronGradient, float* neuronGradient, float* batchMean, float* mean, float* variance, float* std, float* parameters, float* parameterGradient, int height, int width) {
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

void BatchNormalization1D::backPropagate(int num) {
	if (num != 0) {
		prevLayer->backPropagate(num);
		return;
	}
	parameterGradient.constantFill(0);
	float c = (1 - momentum) / batchSize;
	prevLayer->neurons.copyToDevice(0);
	neuronGradient.copyToDevice(2);
	batchMean.copyToDevice(3);
	mean.copyToDevice(4);
	variance.copyToDevice(5);
	std.copyToDevice(6);
	parameters.copyToDevice(7);
	parameterGradient.copyToDevice(8);
	MatrixKernel::runRowKernel(batchSize, size, 0, kernelBackPropagate, c, Matrix2::DEVICES[0][0], Matrix2::DEVICES[0][1], Matrix2::DEVICES[0][2], Matrix2::DEVICES[0][3], Matrix2::DEVICES[0][4], Matrix2::DEVICES[0][5], Matrix2::DEVICES[0][6], Matrix2::DEVICES[0][7], Matrix2::DEVICES[0][8], batchSize, size);
	parameterGradient.copyToHost(8);
	prevLayer->neuronGradient.copyToHost(1);
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
	parameters = Matrix2(FillFunction::UNIT_NORMAL_FILL, 2, size, 0);
	for (int i = 0; i < size; i++) {
		parameters(1, i) = mean1Fill(1, i);
	}
	mean = Matrix2(1, size, 0);
	batchMean = Matrix2(1, size, 0);
	variance = Matrix2(1, size, 0);
	batchVariance = Matrix2(1, size, 0);
	std = Matrix2(1, size, 0);
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
	batchNormalization->variance.sqrt(batchNormalization->std);
}

void BatchNormalization1D::predict(int num) {
	prevLayer->neurons.copyToDevice(0);
	mean.copyToDevice(2);
	std.copyToDevice(3);
	parameters.copyToDevice(4);
	MatrixKernel::runElementKernel(batchSize, size, 0, kernelParameterNormalize, Matrix2::DEVICES[0][0], Matrix2::DEVICES[0][1], Matrix2::DEVICES[0][2], Matrix2::DEVICES[0][3], Matrix2::DEVICES[0][4], batchSize, batchSize * size);
	neurons.copyToHost(1, batchSize * size);
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