// Compile with CUDA

#include "Activation.h"


string* ActivationType::NAMES = new string[9]{ "None", "Sigmoid", "Relu", "Elu", "Selu", "Loglu", "Tanh", "Swish", "Softmax" };
Activation* Activation::NONE = new Activation(ActivationType::NONE);
Activation* Activation::SIGMOID = new Activation(ActivationType::SIGMOID);
Activation* Activation::RELU = new Activation(ActivationType::RELU);
Activation* Activation::ELU = new Activation(ActivationType::ELU);
Activation* Activation::SELU = new Activation(ActivationType::SELU);
Activation* Activation::LOGLU = new Activation(ActivationType::LOGLU);
Activation* Activation::TANH = new Activation(ActivationType::TANH);
Activation* Activation::SWISH = new Activation(ActivationType::SWISH);
Activation* Activation::SOFTMAX = new Activation(ActivationType::SOFTMAX);

Activation::Activation(int activationType, float alpha) {
	this->activationType = activationType;
	this->alpha = alpha;
}

void Activation::save(ofstream& file) {
	file << ActivationType::NAMES[activationType] << "," << alpha;
}

__global__
void kernelSigmoid(float* input, float* output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		output[i] = 1.0 / (1.0 + exp(-input[i]));
	}
}

template<>
bool SigmoidOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	int N = this->A->length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelSigmoid <<< numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], N);
	return true;
}

__global__
void kernelSigmoidBatched(float** input, float** output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		output[batch][i] = 1.0 / (1.0 + exp(-input[batch][i]));
	}
}

template<>
bool SigmoidOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	int N = this->A->length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, A->batchSize);
	kernelSigmoidBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->devices[id][0], queue->devices[id][1], N);
	return true;
}

__global__
void kernelSigmoidDifferentiate(float* input, float* output, float* inputGrad, float* outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		inputGrad[i] = outputGrad[i] * output[i] * (1 - output[i]);
	}
}

template<>
bool SigmoidDifOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelSigmoidDifferentiate << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][3][0], queue->hostDevices[id][2][0], N);
	return true;
}

__global__
void kernelSigmoidDifferentiateBatched(float** input, float** output, float** inputGrad, float** outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		inputGrad[batch][i] = outputGrad[batch][i] * output[batch][i] * (1 - output[batch][i]);
	}
}

template<>
bool SigmoidDifOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, this->in[0]->batchSize);
	kernelSigmoidDifferentiateBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (queue->devices[id][0], queue->devices[id][1], queue->devices[id][3], queue->devices[id][2], N);
	return true;
}

__global__
void kernelRelu(float* input, float* output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		output[i] = input[i] < 0 ? 0 : input[i];
	}
}

template<>
bool ReluOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	int N = this->A->length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelRelu << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], N);
	return true;
}

__global__
void kernelReluBatched(float** input, float** output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		output[batch][i] = input[batch][i] < 0 ? 0 : input[batch][i];
	}
}

template<>
bool ReluOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	int N = this->A->length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, A->batchSize);
	kernelReluBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->devices[id][0], queue->devices[id][1], N);
	return true;
}

__global__
void kernelReluDifferentiate(float* input, float* output, float* inputGrad, float* outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		inputGrad[i] = output[i] > 0 ? outputGrad[i] : 0;
	}
}

template<>
bool ReluDifOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelReluDifferentiate << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][3][0], queue->hostDevices[id][2][0], N);
	return true;
}

__global__
void kernelReluDifferentiateBatched(float** input, float** output, float** inputGrad, float** outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		inputGrad[batch][i] = output[batch][i] > 0 ? outputGrad[batch][i] : 0;
	}
}

template<>
bool ReluDifOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, this->in[0]->batchSize);
	kernelReluDifferentiateBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (queue->devices[id][0], queue->devices[id][1], queue->devices[id][3], queue->devices[id][2], N);
	return true;
}

__global__
void kernelElu(float alpha, float* input, float* output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float value = input[i];
		output[i] = value < 0 ? (alpha * (exp(value) - 1)) : value;
	}
}

template<>
bool EluOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	int N = this->A->length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelElu << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (alpha, queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], N);
	return true;
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

template<>
bool EluOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	int N = this->A->length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, A->batchSize);
	kernelEluBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (alpha, queue->devices[id][0], queue->devices[id][1], N);
	return true;
}

__global__
void kernelEluDifferentiate(float alpha, float* input, float* output, float* inputGrad, float* outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float value = output[i];
		inputGrad[i] = value < 0 ? (outputGrad[i] * (value + alpha)) : (outputGrad[i]);
	}
}

template<>
bool EluDifOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelEluDifferentiate << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (alpha, queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][3][0], queue->hostDevices[id][2][0], N);
	return true;
}

__global__
void kernelEluDifferentiateBatched(float alpha, float** input, float** output, float** inputGrad, float** outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		float value = output[batch][i];
		inputGrad[batch][i] = value < 0 ? (outputGrad[batch][i] * (value + 1.6733 * 1.0507)) : (outputGrad[batch][i] * 1.0507);
	}
}

template<>
bool EluDifOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, this->in[0]->batchSize);
	kernelEluDifferentiateBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (alpha, queue->devices[id][0], queue->devices[id][1], queue->devices[id][3], queue->devices[id][2], N);
	return true;
}

__global__
void kernelSelu(float* input, float* output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float value = input[i];
		output[i] = value < 0 ? (1.6733 * 1.0507 * (exp(value) - 1)) : (1.0507 * value);
	}
}

template<>
bool SeluOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	int N = this->A->length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelSelu << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], N);
	return true;
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

template<>
bool SeluOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	int N = this->A->length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, A->batchSize);
	kernelSeluBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->devices[id][0], queue->devices[id][1], N);
	return true;
}

__global__
void kernelSeluDifferentiate(float* input, float* output, float* inputGrad, float* outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float value = output[i];
		inputGrad[i] = value < 0 ? (outputGrad[i] * (value + 1.6733 * 1.0507)) : (outputGrad[i] * 1.0507);
	}
}

template<>
bool SeluDifOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelSeluDifferentiate << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][3][0], queue->hostDevices[id][2][0], N);
	return true;
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

template<>
bool SeluDifOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, this->in[0]->batchSize);
	kernelSeluDifferentiateBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (queue->devices[id][0], queue->devices[id][1], queue->devices[id][3], queue->devices[id][2], N);
	return true;
}

__global__
void kernelLoglu(float alpha, float* input, float* output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float value = input[i];
		output[i] = value < 0 ? -log(-alpha * value + 1) : value;
	}
}

template<>
bool LogluOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	int N = this->A->length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelLoglu << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (alpha, queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], N);
	return true;
}

__global__
void kernelLogluBatched(float alpha, float** input, float** output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		float value = input[batch][i];
		output[batch][i] = value < 0 ? -log(-alpha * value + 1) : value;
	}
}

template<>
bool LogluOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	int N = this->A->length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, A->batchSize);
	kernelLogluBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (alpha, queue->devices[id][0], queue->devices[id][1], N);
	return true;
}

__global__
void kernelLogluDifferentiate(float alpha, float* input, float* output, float* inputGrad, float* outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float value = output[i];
		inputGrad[i] = value < 0 ? (outputGrad[i] * alpha / (-alpha * value + 1)) : outputGrad[i];
	}
}

template<>
bool LogluDifOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelLogluDifferentiate << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (alpha, queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][3][0], queue->hostDevices[id][2][0], N);
	return true;
}

__global__
void kernelLogluDifferentiateBatched(float alpha, float** input, float** output, float** inputGrad, float** outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		float value = output[batch][i];
		inputGrad[batch][i] = value < 0 ? (outputGrad[batch][i] * alpha / (-alpha * value + 1)) : outputGrad[batch][i];
	}
}

template<>
bool LogluDifOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, this->in[0]->batchSize);
	kernelLogluDifferentiateBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (alpha, queue->devices[id][0], queue->devices[id][1], queue->devices[id][3], queue->devices[id][2], N);
	return true;
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

template<>
bool TanhOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	int N = this->A->length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelTanh << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], N);
	return true;
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

template<>
bool TanhOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	int N = this->A->length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, A->batchSize);
	kernelTanhBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->devices[id][0], queue->devices[id][1], N);
	return true;
}

__global__
void kernelTanhDifferentiate(float* input, float* output, float* inputGrad, float* outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float value = output[i];
		inputGrad[i] = outputGrad[i] * (1 - value * value);
	}
}

template<>
bool TanhDifOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelTanhDifferentiate << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][3][0], queue->hostDevices[id][2][0], N);
	return true;
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

template<>
bool TanhDifOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, this->in[0]->batchSize);
	kernelSigmoidDifferentiateBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (queue->devices[id][0], queue->devices[id][1], queue->devices[id][3], queue->devices[id][2], N);
	return true;
}

__global__
void kernelSwish(float alpha, float* input, float* output, int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float value = input[i];
		output[i] = value / (1 + exp(-value / alpha));
	}
}

template<>
bool SwishOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	int N = this->A->length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelSwish << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (alpha, queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], N);
	return true;
}

__global__
void kernelSwishBatched(float alpha, float** input, float** output, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		float value = input[batch][i];
		output[batch][i] = value / (1 + exp(-value / alpha));
	}
}

template<>
bool SwishOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	int N = this->A->length;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, this->A->batchSize);
	kernelSwishBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (alpha, queue->devices[id][0], queue->devices[id][1], N);
	return true;
}

__global__
void kernelSwishDifferentiate(float alpha, float* input, float* output, float* inputGrad, float* outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float in = input[i];
		float out = output[i];
		inputGrad[i] = in == 0 ? (0.5 * outputGrad[i]) : (outputGrad[i] * out * (1 + (in - out) / alpha) / in);
	}
}

template<>
bool SwishDifOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelSwishDifferentiate << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (alpha, queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][3][0], queue->hostDevices[id][2][0], N);
	return true;
}

__global__
void kernelSwishDifferentiateBatched(float alpha, float** input, float** output, float** inputGrad, float** outputGrad, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		float in = input[batch][i];
		float out = output[batch][i];
		inputGrad[batch][i] = in == 0 ? (0.5 * outputGrad[batch][i]) : (outputGrad[batch][i] * out * (1 + (in - out) / alpha) / in);
	}
}

template<>
bool SwishDifOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, this->in[0]->batchSize);
	kernelSwishDifferentiateBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (this->alpha, queue->devices[id][0], queue->devices[id][1], queue->devices[id][3], queue->devices[id][2], N);
	return true;
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

template<>
bool SoftmaxOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	int N = this->A->length;
	kernelSoftmax << < this->A->height, Utils::THREADS_PER_BLOCK, Utils::THREADS_PER_BLOCK * sizeof(float), queue->streams[id] >> > (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], this->A->height, this->A->width);
	return true;
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

template<>
bool SoftmaxOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	int N = this->A->length;
	dim3 blocks(this->A->height, this->A->batchSize);
	kernelSoftmaxBatched <<< blocks, Utils::THREADS_PER_BLOCK, Utils::THREADS_PER_BLOCK * sizeof(float), queue->streams[id] >> > (queue->devices[id][0], queue->devices[id][1], this->A->height, this->A->width);
	return true;
}

__global__
void kernelSoftmaxDifferentiate(float* input, float* output, float* inputGradient, float* outputGradient, int height, int width) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < height * width) {
		int i = id % height;
		int j = id / height;
		float sum = 0.0f;
		for (int k = 0; k < width; k++) {
			sum += outputGradient[i + height * k] * ((j == k ? 1 : 0) - output[i + height * k]);
		}
		inputGradient[i + height * j] = output[i + height * j] * sum;
	}
}

template<>
bool SoftmaxDifOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelSoftmaxDifferentiate << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][3][0], queue->hostDevices[id][2][0], this->in[0]->height, this->in[0]->width);
	return true;
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

template<>
bool SoftmaxDifOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, this->in[0]->batchSize);
	kernelSoftmaxDifferentiateBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (queue->devices[id][0], queue->devices[id][1], queue->devices[id][3], queue->devices[id][2], this->in[0]->height, this->in[0]->width);
	return true;
}