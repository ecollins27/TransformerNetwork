#include "Activation.h"
#include "MatrixKernel.h"

Activation* Activation::NONE = new None();
Activation* Activation::SIGMOID = new Sigmoid();
Activation* Activation::RELU = new Relu();
Activation* Activation::ELU = new Elu(1);
Activation* Activation::SELU = new Selu();
Activation* Activation::TANH = new Tanh();
Activation* Activation::SWISH = new Swish();
Activation* Activation::SOFTMAX = new Softmax();
Activation* Activation::ALL_ACTIVATIONS[Activation::NUM_ACTIVATIONS] = {NONE, SIGMOID, RELU, ELU, SELU, TANH, SWISH, SOFTMAX };

void None::operate(Matrix2& input, Matrix2& output) {
	output.copy(input);
}

void None::operate(MatrixBatch& input, MatrixBatch& output) {
	output.copy(input);
}

void None::differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGrad, Matrix2& outputGrad) {
	inputGrad.copy(outputGrad);
}

void None::differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGrad, MatrixBatch& outputGrad) {
	inputGrad.copy(outputGrad);
}

Activation* None::clone() {
	return new None();
}

__global__
void kernelSigmoid(float* input, float* output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		output[i] = 1.0 / (1.0 + exp(-input[i]));
	}
}

void Sigmoid::operate(Matrix2& input, Matrix2& output) {
	MatrixKernel::runElementKernel(input.height, input.width, 0, kernelSigmoid, input.device, output.device, input.length);
	output.copyToHost();
}

__global__
void kernelSigmoidBatched(float** input, float** output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		output[batch][i] = 1.0 / (1.0 + exp(-input[batch][i]));
	}
}

void Sigmoid::operate(MatrixBatch& input, MatrixBatch& output) {
	MatrixKernel::runElementKernelBatched(input.batchSize, input.height, input.width, 0, kernelSigmoidBatched, input.device, output.device, input.length);
	output.copyToHost();
}

__global__
void kernelSigmoidDifferentiate(float* input, float* output, float* inputGrad, float* outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		inputGrad[i] = outputGrad[i] * output[i] * (1 - output[i]);
	}
}

void Sigmoid::differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGrad, Matrix2& outputGrad) {
	MatrixKernel::runElementKernel(input.height, input.width, 0, kernelSigmoidDifferentiate, input.device, output.device, inputGrad.device, outputGrad.device, input.length);
	inputGrad.copyToHost();
}

__global__
void kernelSigmoidDifferentiateBatched(float** input, float** output, float** inputGrad, float** outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		inputGrad[batch][i] = outputGrad[batch][i] * output[batch][i] * (1 - output[batch][i]);
	}
}

void Sigmoid::differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGrad, MatrixBatch& outputGrad) {
	MatrixKernel::runElementKernelBatched(input.batchSize, input.height, input.width, 0, kernelSigmoidDifferentiateBatched, input.device, output.device, inputGrad.device, outputGrad.device, input.length);
	inputGrad.copyToHost();
}

Activation* Sigmoid::clone() {
	return new Sigmoid();
}

__global__
void kernelRelu(float* input, float* output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		output[i] = input[i] < 0 ? 0 : input[i];
	}
}

void Relu::operate(Matrix2& input, Matrix2& output) {
	MatrixKernel::runElementKernel(input.height, input.width, 0, kernelRelu, input.device, output.device, input.length);
	output.copyToHost();
}

__global__
void kernelReluBatched(float** input, float** output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		output[batch][i] = input[batch][i] < 0 ? 0 : input[batch][i];
	}
}

void Relu::operate(MatrixBatch& input, MatrixBatch& output) {
	MatrixKernel::runElementKernelBatched(input.batchSize, input.height, input.width, 0, kernelReluBatched, input.device, output.device, input.length);
	output.copyToHost();
}

__global__
void kernelReluDifferentiate(float* input, float* output, float* inputGrad, float* outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		inputGrad[i] = output[i] > 0 ? outputGrad[i] : 0;
	}
}

void Relu::differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGrad, Matrix2& outputGrad) {
	MatrixKernel::runElementKernel(input.height, input.width, 0, kernelReluDifferentiate, input.device, output.device, inputGrad.device, outputGrad.device, input.length);
	inputGrad.copyToHost();
}

__global__
void kernelReluDifferentiateBatched(float** input, float** output, float** inputGrad, float** outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		inputGrad[batch][i] = output[batch][i] > 0 ? outputGrad[batch][i] : 0;
	}
}

void Relu::differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGrad, MatrixBatch& outputGrad) {
	MatrixKernel::runElementKernelBatched(input.batchSize, input.height, input.width, 0, kernelReluDifferentiateBatched, input.device, output.device, inputGrad.device, outputGrad.device, input.length);
	inputGrad.copyToHost();
}

Activation* Relu::clone() {
	return new Relu();
}

Elu::Elu(float alpha) {
	this->alpha = alpha;
}

__global__
void kernelElu(float alpha, float* input, float* output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float value = input[i];
		output[i] = value < 0 ? (alpha * (exp(value) - 1)) : value;
	}
}

void Elu::operate(Matrix2& input, Matrix2& output) {
	MatrixKernel::runElementKernel(input.height, input.width, 0, kernelElu, alpha, input.device, output.device, input.length);
	output.copyToHost();
}

__global__
void kernelEluBatched(float alpha, float** input, float** output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		float value = input[batch][i];
		output[batch][i] = value < 0 ? (alpha * (exp(value) - 1)) : value;
	}
}

void Elu::operate(MatrixBatch& input, MatrixBatch& output) {
	MatrixKernel::runElementKernelBatched(input.batchSize, input.height, input.width, 0, kernelEluBatched, alpha, input.device, output.device, input.length);
	output.copyToHost();
}

__global__
void kernelEluDifferentiate(float alpha, float* input, float* output, float* inputGrad, float* outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float value = output[i];
		inputGrad[i] = value < 0 ? (outputGrad[i] * (value + alpha)) : outputGrad[i];
	}
}

void Elu::differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGrad, Matrix2& outputGrad) {
	MatrixKernel::runElementKernel(input.height, input.width, 0, kernelEluDifferentiate, alpha, input.device, output.device, inputGrad.device, outputGrad.device, input.length);
	inputGrad.copyToHost();
}

__global__
void kernelEluDifferentiateBatched(float alpha, float** input, float** output, float** inputGrad, float** outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		float value = output[batch][i];
		inputGrad[batch][i] = value < 0 ? (outputGrad[batch][i] * (value + alpha)) : outputGrad[batch][i];
	}
}

void Elu::differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGrad, MatrixBatch& outputGrad) {
	MatrixKernel::runElementKernelBatched(input.batchSize, input.height, input.width, 0, kernelEluDifferentiateBatched, alpha, input.device, output.device, inputGrad.device, outputGrad.device, input.length);
	inputGrad.copyToHost();
}

Activation* Elu::clone() {
	return new Elu(alpha);
}

void Elu::save(ofstream& file) {
	file << "Elu" << "," << alpha << ",";
}

__global__
void kernelSelu(float* input, float* output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float value = input[i];
		output[i] = value < 0 ? (1.6733 * 1.0507 * (exp(value) - 1)) : (1.0507 * value);
	}
}

void Selu::operate(Matrix2& input, Matrix2& output) {
	MatrixKernel::runElementKernel(input.height, input.width, 0, kernelSelu, input.device, output.device, input.length);
	output.copyToHost();
}

__global__
void kernelSeluBatched(float** input, float** output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		float value = input[batch][i];
		output[batch][i] = value < 0 ? (1.6733 * 1.0507 * (exp(value) - 1)) : (1.0507 * value);
	}
}

void Selu::operate(MatrixBatch& input, MatrixBatch& output) {
	MatrixKernel::runElementKernelBatched(input.batchSize, input.height, input.width, 0, kernelSeluBatched, input.device, output.device, input.length);
	output.copyToHost();
}

__global__
void kernelSeluDifferentiate(float* input, float* output, float* inputGrad, float* outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float value = output[i];
		inputGrad[i] = value < 0 ? (outputGrad[i] * (value + 1.6733 * 1.0507)) : (outputGrad[i] * 1.0507);
	}
}

void Selu::differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGrad, Matrix2& outputGrad) {
	MatrixKernel::runElementKernel(input.height, input.width, 0, kernelSeluDifferentiate, input.device, output.device, inputGrad.device, outputGrad.device, input.length);
	inputGrad.copyToHost();
}

__global__
void kernelSeluDifferentiateBatched(float** input, float** output, float** inputGrad, float** outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		float value = output[batch][i];
		inputGrad[batch][i] = value < 0 ? (outputGrad[batch][i] * (value + 1.6733 * 1.0507)) : (outputGrad[batch][i] * 1.0507);
	}
}

void Selu::differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGrad, MatrixBatch& outputGrad) {
	MatrixKernel::runElementKernelBatched(input.batchSize, input.height, input.width, 0, kernelSeluDifferentiateBatched, input.device, output.device, inputGrad.device, outputGrad.device, input.length);
	inputGrad.copyToHost();
}

Activation* Selu::clone() {
	return new Selu();
}

__global__
void kernelTanh(float* input, float* output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float value = input[i];
		float eX = exp(value);
		float eNegX = exp(-value);
		output[i] = (eX - eNegX) / (eX + eNegX);
	}
}

void Tanh::operate(Matrix2& input, Matrix2& output) {
	MatrixKernel::runElementKernel(input.height, input.width, 0, kernelTanh, input.device, output.device, input.length);
	output.copyToHost();
}

__global__
void kernelTanhBatched(float** input, float** output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		float value = input[batch][i];
		float eX = exp(value);
		float eNegX = exp(-value);
		output[batch][i] = (eX - eNegX) / (eX + eNegX);
	}
}

void Tanh::operate(MatrixBatch& input, MatrixBatch& output) {
	MatrixKernel::runElementKernelBatched(input.batchSize, input.height, input.width, 0, kernelTanhBatched, input.device, output.device, input.length);
	output.copyToHost();
}

__global__
void kernelTanhDifferentiate(float* input, float* output, float* inputGrad, float* outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float value = output[i];
		inputGrad[i] = outputGrad[i] * (1 - value * value);
	}
}

void Tanh::differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGrad, Matrix2& outputGrad) {
	MatrixKernel::runElementKernel(input.height, input.width, 0, kernelTanhDifferentiate, input.device, output.device, inputGrad.device, outputGrad.device, input.length);
	inputGrad.copyToHost();
}

__global__
void kernelTanhDifferentiateBatched(float** input, float** output, float** inputGrad, float** outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		float value = output[batch][i];
		inputGrad[batch][i] = outputGrad[batch][i] * (1 - value * value);
	}
}

void Tanh::differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGrad, MatrixBatch& outputGrad) {
	MatrixKernel::runElementKernelBatched(input.batchSize, input.height, input.width, 0, kernelTanhDifferentiateBatched, input.device, output.device, inputGrad.device, outputGrad.device, input.length);
	inputGrad.copyToHost();
}

Activation* Tanh::clone() {
	return new Tanh();
}

__global__
void kernelSwish(float* input, float* output, int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float value = input[i];
		output[i] = value / (1 + exp(-value));
	}
}

void Swish::operate(Matrix2& input, Matrix2& output) {
	MatrixKernel::runElementKernel(input.height, input.width, 0, kernelSwish, input.device, output.device, input.length);
	output.copyToHost();
}

__global__
void kernelSwishBatched(float** input, float** output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		float value = input[batch][i];
		output[batch][i] = value / (1 + exp(-value));
	}
}

void Swish::operate(MatrixBatch& input, MatrixBatch& output) {
	MatrixKernel::runElementKernelBatched(input.batchSize, input.height, input.width, 0, kernelSwishBatched, input.device, output.device, input.length);
	output.copyToHost();
}

__global__
void kernelSwishDifferentiate(float* input, float* output, float* inputGrad, float* outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float in = input[i];
		float out = output[i];
		inputGrad[i] = in == 0 ? (0.5 * outputGrad[i]) : (outputGrad[i] * out * (in - out + 1) / in);
	}
}

void Swish::differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGrad, Matrix2& outputGrad) {
	MatrixKernel::runElementKernel(input.height, input.width, 0, kernelSwishDifferentiate, input.device, output.device, inputGrad.device, outputGrad.device, input.length);
	inputGrad.copyToHost();
}

__global__
void kernelSwishDifferentiateBatched(float** input, float** output, float** inputGrad, float** outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		float in = input[batch][i];
		float out = output[batch][i];
		inputGrad[batch][i] = in == 0 ? (0.5 * outputGrad[batch][i]) : (outputGrad[batch][i] * out * (in - out + 1) / in);
	}
}

void Swish::differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGrad, MatrixBatch& outputGrad) {
	MatrixKernel::runElementKernelBatched(input.batchSize, input.height, input.width, 0, kernelSwishDifferentiateBatched, input.device, output.device, inputGrad.device, outputGrad.device, input.length);
	inputGrad.copyToHost();
}

Activation* Swish::clone() {
	return new Swish();
}

__global__
void kernelSoftmax(float* input, float* output, int height, int width) {
	extern __shared__ float shared[];

	int row = blockIdx.x;
	int tid = threadIdx.x;
	int stride = blockDim.x;

	float maxValue = input[row + height * tid];
	for (int i = tid + stride; i < width; i += stride) {
		maxValue = max(maxValue, input[row + height * i]);
	}

	shared[tid] = maxValue;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shared[tid] = max(shared[tid], shared[tid + s]);
		}
		__syncthreads();
	}

	maxValue = shared[0];

	float sum = 0;
	for (int i = tid; i < width; i += stride) {
		output[row + height * i] = exp(input[row + height * i] - maxValue + 10);
		sum += output[row + height * i];
	}
	shared[tid] = sum;

	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shared[tid] += shared[tid + s];
		}
		__syncthreads();
	}

	for (int i = tid; i < width; i += stride) {
		output[row + height * i] /= shared[0];
	}
}

void Softmax::operate(Matrix2& input, Matrix2& output) {
	MatrixKernel::runRowKernel(input.height, input.width, Matrix2::THREADS_PER_BLOCK * sizeof(float), kernelSoftmax, input.device, output.device, input.height, input.width);
	output.copyToHost();
}

__global__
void kernelSoftmaxBatched(float** input, float** output, int height, int width) {
	extern __shared__ float shared[];

	int row = blockIdx.x;
	int batch = blockIdx.y;
	int tid = threadIdx.x;
	int stride = blockDim.x;

	float maxValue = input[batch][row + height * tid];
	for (int i = tid + stride; i < width; i += stride) {
		maxValue = max(maxValue, input[batch][row + height * i]);
	}

	shared[tid] = maxValue;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shared[tid] = max(shared[tid], shared[tid + s]);
		}
		__syncthreads();
	}

	maxValue = shared[0];

	float sum = 0;
	for (int i = tid; i < width; i += stride) {
		output[batch][row + height * i] = exp(input[batch][row + height * i] - maxValue + 10);
		sum += output[batch][row + height * i];
	}
	shared[tid] = sum;

	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shared[tid] += shared[tid + s];
		}
		__syncthreads();
	}

	for (int i = tid; i < width; i += stride) {
		output[batch][row + height * i] /= shared[0];
	}
}

void Softmax::operate(MatrixBatch& input, MatrixBatch& output) {
	MatrixKernel::runRowKernelBatched(input.batchSize, input.height, input.width, Matrix2::THREADS_PER_BLOCK * sizeof(float), kernelSoftmaxBatched, input.device, output.device, input.height, input.width);
	output.copyToHost();
}

__global__
void kernelSoftmaxDifferentiate(float* input, float* output, float* inputGradient, float* outputGradient, int height, int width) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < height * width) {
		int i = id % height;
		int j = id / height;
		float sum = 0.0f;
		for (int k = 0; k < width; k++) {
			sum += outputGradient[i + height * k] * output[i + height * j] * ((j == k ? 1 : 0) - output[i + height * k]);
		}
		inputGradient[i + height * j] = sum;
	}
}

void Softmax::differentiate(Matrix2& input, Matrix2& output, Matrix2& inputGrad, Matrix2& outputGrad) {
	MatrixKernel::runElementKernel(input.height, input.width, 0, kernelSoftmaxDifferentiate, input.device, output.device, inputGrad.device, outputGrad.device, input.height, input.width);
	inputGrad.copyToHost();
}

__global__
void kernelSoftmaxDifferentiateBatched(float** input, float** output, float** inputGradient, float** outputGradient, int height, int width) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (id < height * width) {
		int i = id % height;
		int j = id / height;
		float sum = 0.0f;
		for (int k = 0; k < width; k++) {
			sum += outputGradient[batch][i + height * k] * output[batch][i + height * j] * ((j == k ? 1 : 0) - output[batch][i + height * k]);
		}
		inputGradient[batch][i + height * j] = sum;
	}
}

void Softmax::differentiate(MatrixBatch& input, MatrixBatch& output, MatrixBatch& inputGrad, MatrixBatch& outputGrad) {
	MatrixKernel::runElementKernelBatched(input.batchSize, input.height, input.width, 0, kernelSoftmaxDifferentiateBatched, input.device, output.device, inputGrad.device, outputGrad.device, input.width, input.length);
	inputGrad.copyToHost();
}

Activation* Softmax::clone() {
	return new Softmax();
}

bool Softmax::isDiagonal() {
	return false;
}