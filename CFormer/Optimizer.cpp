// Compile with CUDA

#include "Optimizer.h"

template<>
Optimizer<>* Optimizer<>::GRADIENT_DESCENT = new GradientDescent<>(0);
template<>
Optimizer<>* Optimizer<>::MOMENTUM = new Momentum<>(0.9, 0);
template<>
Optimizer<>* Optimizer<>::ADAM = new Adam<>(0.9,0.999, 0);
template<>
Optimizer<>* Optimizer<>::ADEMAMIX = new AdEMAMix<>(0.9, 0.9999, 0.999, 5, 0);

template<typename Type>
GradientDescent<Type>::GradientDescent(float regConstant) {
	this->regConstant = this->regConstant;
}

template<typename Type>
void GradientDescent<Type>::initApplicationQueue(OperationQueue& queue, Type& weights, float learningRate, int batchSize, int& t) {
	queue.enqueue(new Scale(this->weightGradient, 1.0 / batchSize));
	if (this->regConstant != 0) {
		queue.enqueue(new LinearCombo(1, this->weightGradient, 2 * this->regConstant, weights, this->weightGradient));
	}
	queue.enqueue(new LinearCombo(1, weights, -learningRate, this->weightGradient, weights));
	queue.enqueue(new ConstantFill(this->weightGradient, 0));
}

template<>
void GradientDescent<Matrix>::setDimensions(int batchSize, int height, int width) {
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	this->weightGradient = Matrix(height, width);
}

template<>
void GradientDescent<MatrixBatch>::setDimensions(int batchSize, int height, int width) {
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	this->weightGradient = MatrixBatch(batchSize, height, width);
}

template<typename Type>
Momentum<Type>::Momentum(float beta, float regConstant) {
	this->beta = beta;
	this->regConstant = regConstant;
}

template<typename Type>
void Momentum<Type>::initApplicationQueue(OperationQueue& queue, Type& weights, float learningRate, int batchSize, int& t) {
	queue.enqueue(new Scale(this->weightGradient, 1.0 / batchSize));
	if (this->regConstant != 0) {
		queue.enqueue(new LinearCombo(1, this->weightGradient, 2 * this->regConstant, weights, this->weightGradient));
	}
	queue.enqueue(new LinearCombo(beta, M, -learningRate, this->weightGradient, M));
	queue.enqueue(new Add(weights, M, weights));
	queue.enqueue(new ConstantFill(this->weightGradient, 0));
}

template<>
void Momentum<Matrix>::setDimensions(int batchSize, int height, int width) {
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	M = Matrix(height, width);
	this->weightGradient = Matrix(height, width);
}

template<>
void Momentum<MatrixBatch>::setDimensions(int batchSize, int height, int width) {
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	M = MatrixBatch(batchSize, height, width);
	this->weightGradient = MatrixBatch(batchSize, height, width);
}

template<typename Type>
Adam<Type>::Adam(float beta1, float beta2, float regConstant) {
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->regConstant = regConstant;
}

template<typename Type>
void Adam<Type>::initApplicationQueue(OperationQueue& queue, Type& weights, float learningRate, int batchSize, int& t) {
	float mScalar = 1.0 / (1 - pow(beta1, t));
	float sScalar = 1.0 / (1 - pow(beta2, t));
	queue.enqueue(new Scale(this->weightGradient, 1.0 / batchSize));
	if (this->regConstant != 0) {
		queue.enqueue(new LinearCombo(1, this->weightGradient, 2 * this->regConstant, weights, this->weightGradient));
	}
	queue.enqueue(new LinearCombo(beta1, M, 1 - beta1, this->weightGradient, M));
	queue.enqueue(new LinearCombo2(beta2, S, 1 - beta2, this->weightGradient, S));
	queue.enqueue(new AdamOperation(weights, M, S, beta1, beta2, learningRate, t));
	queue.enqueue(new ConstantFill(this->weightGradient, 0));
}

template<>
void Adam<Matrix>::setDimensions(int batchSize, int height, int width) {
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	M = Matrix(height, width);
	S = Matrix(height, width);
	this->weightGradient = Matrix(height, width);
}

template<>
void Adam<MatrixBatch>::setDimensions(int batchSize, int height, int width) {
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	M = MatrixBatch(batchSize, height, width);
	S = MatrixBatch(batchSize, height, width);
	this->weightGradient = MatrixBatch(batchSize, height, width);
}

template<typename Type>
AdEMAMix<Type>::AdEMAMix(float beta1, float beta2, float beta3, float alpha, float regConstant) {
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->beta3 = beta3;
	this->alpha = alpha;
	this->regConstant = regConstant;
}

template<typename Type>
void AdEMAMix<Type>::initApplicationQueue(OperationQueue& queue, Type& weights, float learningRate, int batchSize, int& t) {
	float mScalar = 1.0 / (1 - pow(beta1, t));
	float sScalar = 1.0 / (1 - pow(beta2, t));
	queue.enqueue(new Scale(this->weightGradient, 1.0 / batchSize));
	if (this->regConstant != 0) {
		queue.enqueue(new LinearCombo(1, this->weightGradient, 2 * this->regConstant, weights, this->weightGradient));
	}
	queue.enqueue(new LinearCombo(beta1, M1, 1 - beta1, this->weightGradient, M1));
	queue.enqueue(new LinearCombo(beta3, M2, 1 - beta3, this->weightGradient, M2));
	queue.enqueue(new LinearCombo2(beta2, S, 1 - beta2, this->weightGradient, S));
	queue.enqueue(new AdEMAMixOperation(weights, M1, M2, S, beta1, beta2, beta3, learningRate, alpha, t));
	queue.enqueue(new ConstantFill(this->weightGradient, 0));
}

template<>
void AdEMAMix<Matrix>::setDimensions(int batchSize, int height, int width) {
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	M1 =Matrix(height, width);
	M2 = Matrix(height, width);
	S = Matrix(height, width);
	this->weightGradient = Matrix(height, width);
}

template<>
void AdEMAMix<MatrixBatch>::setDimensions(int batchSize, int height, int width) {
	this->batchSize = batchSize;
	this->height = height;
	this->width = width;
	M1 = MatrixBatch(batchSize, height, width);
	M2 = MatrixBatch(batchSize, height, width);
	S = MatrixBatch(batchSize, height, width);
	this->weightGradient = MatrixBatch(batchSize, height, width);
}

__global__
void kernelLinearCombo2(int N, float c1, float* A, float c2, float* B, float* C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = c1 * A[i] + c2 * B[i] * B[i];
	}
}

template<>
bool LinearCombo2<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = A->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelLinearCombo2 << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (N, c1, queue->hostDevices[id][0][0], c2, queue->hostDevices[id][1][0], queue->hostDevices[id][2][0]);
	return true;
}

__global__
void kernelLinearCombo2Batched(float c1, float** A, float c2, float** B, float** C, int N) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;

	if (n < N) {
		C[batch][n] = c1 * A[batch][n] + c2 * B[batch][n] * B[batch][n];
	}
}

template<>
bool LinearCombo2<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = A->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, A->batchSize);
	kernelLinearCombo2Batched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (c1, queue->devices[id][0], c2, queue->devices[id][1], queue->devices[id][2], N);
	return true;
}

__global__
void kernelAdam(float* weights, float* M, float* S, float learningRate, float mScalar, float sScalar, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		weights[i] -= learningRate * (M[i] * mScalar) / sqrt((S[i] * sScalar) + 0.0000001);
	}
}

template<>
bool AdamOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	float mScalar = 1.0 / (1 - pow(this->beta1, *t));
	float sScalar = 1.0 / (1 - pow(this->beta2, *t));
	kernelAdam <<< numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][2][0], this->learningRate, mScalar, sScalar, N);
	return true;
}

__global__
void kernelAdamBatched(float** weights, float** M, float** S, float learningRate, float mScalar, float sScalar, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	if (i < N) {
		weights[j][i] -= learningRate * (M[j][i] * mScalar) / sqrt((S[j][i] * sScalar) + 0.0000001);
	}
}

template<>
bool AdamOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, this->in[0]->batchSize);
	float mScalar = 1.0 / (1 - pow(this->beta1, *t));
	float sScalar = 1.0 / (1 - pow(this->beta2, *t));
	kernelAdamBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->devices[id][0], queue->devices[id][1], queue->devices[id][2], this->learningRate, mScalar, sScalar, N);
	return true;
}

__global__
void kernelAdEMAMix(float* weights, float* M1, float* M2, float* S, float learningRate, float mScalar, float sScalar, float alpha, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		weights[i] -= learningRate * (mScalar * M1[i] + alpha * M2[i]) / sqrt(sScalar * S[i] + 0.0000001);
	}
}

template<>
bool AdEMAMixOperation<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	float mScalar = 1.0 / (1 - pow(this->beta1, *t));
	float sScalar = 1.0 / (1 - pow(this->beta2, *t));
	kernelAdEMAMix << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][2][0], queue->hostDevices[id][3][0], this->learningRate, mScalar, sScalar, this->alpha, N);
	return true;
}

__global__
void kernelAdEMAMixBatched(float** weights, float** M1, float** M2, float** S, float learningRate, float mScalar, float sScalar, float alpha, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	if (i < N) {
		weights[j][i] -= learningRate * (mScalar * M1[j][i] + alpha * M2[j][i]) / sqrt(sScalar * S[j][i] + 0.0000001);
	}
}

template<>
bool AdEMAMixOperation<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = this->in[0]->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, this->in[0]->batchSize);
	float mScalar = 1.0 / (1 - pow(this->beta1, *t));
	float sScalar = 1.0 / (1 - pow(this->beta2, *t));
	kernelAdEMAMixBatched << < blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->devices[id][0], queue->devices[id][1], queue->devices[id][2], queue->devices[id][3], this->learningRate, mScalar, sScalar, this->alpha, N);
	return true;
}