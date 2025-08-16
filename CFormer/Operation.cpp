// Compile with CUDA

#include "Operation.h"
#include "OperationQueue.h"

template<typename T>
void extendArray(T*& array, int oldLength, int newLength) {
	T* newArray;
	cudaError_t err = cudaMallocHost(&newArray, newLength * sizeof(T));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
	if (array != NULL) {
		err = cudaMemcpy(newArray, array, oldLength * sizeof(T), cudaMemcpyHostToHost);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
		}
		err = cudaFreeHost(array);
	}
	array = newArray;
}

template<>
bool HostToDeviceCopy<Matrix>::operate(OperationQueue* queue, int threadID) {
	if (this->threadID->load() == -1){
		bool foundThread = false;
		bool refFalse;
		bool refTrue;
		int refNegOne;
		for (int i = 0; i < queue->numThreads; i++) {
			refTrue = true;
			refNegOne = -1;
			if (queue->deviceLocks[i].compare_exchange_strong(refTrue, false)) {
				if (!this->threadID->compare_exchange_strong(refNegOne, i)) {
					queue->deviceLocks[i].compare_exchange_strong(refFalse, true);
				}
				else {
					queue->devicesUsed.fetch_add(1);
				}
				foundThread = true;
				break;
			}
		}
		if (!foundThread) {
			return false;
		}
	}
	if (batchSize > 1) {
		cudaError_t err;
		for (int i = 0; i < batchSize; i++) {
			err = cudaMemcpy(queue->hostDevices[this->threadID->load()][this->deviceNum][i], A->host, A->length * sizeof(float), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
			}
		}
	}
	else {
		cudaError_t err = cudaMemcpyAsync(queue->hostDevices[this->threadID->load()][this->deviceNum][0], A->host, A->length * sizeof(float), cudaMemcpyHostToDevice, queue->streams[this->threadID->load()]);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
	//printf("Host To Device copy completed.  Operation: %p  Prereq: %p\n", this, prereq);
	return true;
}

template<>
bool HostToDeviceCopy<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	if (this->threadID->load() == -1) {
		bool foundThread = false;
		bool refTrue;
		int refNegOne;
		for (int i = 0; i < queue->numThreads; i++) {
			refTrue = true;
			refNegOne = -1;
			if (queue->deviceLocks[i].compare_exchange_strong(refTrue, false)) {
				this->threadID->compare_exchange_strong(refNegOne, i);
				foundThread = true;
				break;
			}
		}
		if (!foundThread && this->threadID->load() == -1) {
			return false;
		}
	}
	cudaError_t err;
	for (int i = 0; i < A->batchSize; i++) {
		err = cudaMemcpy(queue->hostDevices[this->threadID->load()][this->deviceNum][i], A->host[i], A->length * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
	return true;
}

template<>
void HostToDeviceCopy<Matrix>::applyToStream(OperationQueue* queue) {
	if (queue->numDevices <= deviceNum) {
		extendArray(queue->deviceBatchSizes, queue->numDevices, deviceNum + 1);
		extendArray(queue->deviceLengths, queue->numDevices, deviceNum + 1);
		for (int i = queue->numDevices; i <= deviceNum; i++) {
			queue->deviceBatchSizes[i] = 0;
			queue->deviceLengths[i] = 0;
		}
		queue->numDevices = deviceNum + 1;
	}
	queue->deviceBatchSizes[deviceNum] = max(queue->deviceBatchSizes[deviceNum], 1);
	queue->deviceLengths[deviceNum] = max(queue->deviceLengths[deviceNum], A->maxLength);
}

template<>
void HostToDeviceCopy<MatrixBatch>::applyToStream(OperationQueue* queue) {
	if (queue->numDevices <= deviceNum) {
		extendArray(queue->deviceBatchSizes, queue->numDevices, deviceNum + 1);
		extendArray(queue->deviceLengths, queue->numDevices, deviceNum + 1);
		for (int i = queue->numDevices; i <= deviceNum; i++) {
			queue->deviceBatchSizes[i] = 0;
			queue->deviceLengths[i] = 0;
		}
		queue->numDevices = deviceNum + 1;
	}
	queue->deviceBatchSizes[deviceNum] = max(queue->deviceBatchSizes[deviceNum], A->batchSize);
	queue->deviceLengths[deviceNum] = max(queue->deviceLengths[deviceNum], A->maxLength);
}

__global__
void kernelBiasSet(float* column, int offset, int height) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < height) {
		column[offset + i] = 1.0f;
	}
}

template<>
bool DeviceToHostCopy<Matrix>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID->load();
	if (this->A->isLayerOutput) {
		int N = this->A->height;
		int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
		kernelBiasSet << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->hostDevices[id][this->deviceNum][0], this->A->length, N);
	}
	cudaError_t err = cudaMemcpyAsync(A->host, queue->hostDevices[id][this->deviceNum][0], A->length * sizeof(float), cudaMemcpyDeviceToHost, queue->streams[id]);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
	cudaStreamSynchronize(queue->streams[id]);
	this->outputsCopied->fetch_sub(1);
	if (this->outputsCopied->compare_exchange_strong(this->refZERO, this->numOutputs)) {
		queue->deviceLocks[id].store(true);
		queue->devicesUsed.fetch_sub(1);
	}
	return true;
}

template<>
bool DeviceToHostCopy<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	cudaError_t err;
	for (int i = 0; i < A->batchSize; i++) {
		err = cudaMemcpy(A->host[i], queue->hostDevices[this->threadID->load()][this->deviceNum][i], A->length * sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
	queue->deviceLocks[this->threadID->load()].store(true);
	this->threadID->store(-1);
	return true;
}

template<>
void DeviceToHostCopy<Matrix>::applyToStream(OperationQueue* queue) {
	if (queue->numDevices <= deviceNum) {
		extendArray(queue->deviceBatchSizes, queue->numDevices, deviceNum + 1);
		extendArray(queue->deviceLengths, queue->numDevices, deviceNum + 1);
		for (int i = queue->numDevices; i <= deviceNum; i++) {
			queue->deviceBatchSizes[i] = 0;
			queue->deviceLengths[i] = 0;
		}
		queue->numDevices = deviceNum + 1;
	}
	queue->deviceBatchSizes[deviceNum] = max(queue->deviceBatchSizes[deviceNum], 1);
	queue->deviceLengths[deviceNum] = max(queue->deviceLengths[deviceNum], A->maxLength);
}

template<>
void DeviceToHostCopy<MatrixBatch>::applyToStream(OperationQueue* queue) {
	if (queue->numDevices <= deviceNum) {
		extendArray(queue->deviceBatchSizes, queue->numDevices, deviceNum + 1);
		extendArray(queue->deviceLengths, queue->numDevices, deviceNum + 1);
		for (int i = queue->numDevices; i <= deviceNum; i++) {
			queue->deviceBatchSizes[i] = 0;
			queue->deviceLengths[i] = 0;
		}
		queue->numDevices = deviceNum + 1;
	}
	queue->deviceBatchSizes[deviceNum] = max(queue->deviceBatchSizes[deviceNum], A->batchSize);
	queue->deviceLengths[deviceNum] = max(queue->deviceLengths[deviceNum], A->maxLength);
}

void DOperation::applyToStream(OperationQueue* queue) {
	return;
}

void DOperation::findPrereqs(vector<Operation*> operations, int index) {
	return;
}

template<>
bool Print<Matrix>::operate(OperationQueue* queue, int threadID) {
	for (int i = 0; i < this->A->height; i++) {
		for (int j = 0; j < this->A->isLayerOutput? (this->A->width + 1):this->A->width; j++) {
			printf("%f  ", this->A->host[this->A->e(i, j)]);
		}
		printf("\n");
	}
	printf("\n");
	return true;
}

template<>
bool Print<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	for (int n = 0; n < this->A->batchSize; n++) {
		for (int i = 0; i < this->A->height; i++) {
			for (int j = 0; j < this->A->isLayerOutput ? (this->A->width + 1) : this->A->width; j++) {
				printf("%f  ", this->A->host[n][this->A->e(i, j)]);
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");
	return true;
}

template<>
bool CopyTo<Matrix>::operate(OperationQueue* queue, int threadID) {
	cudaError_t err = cudaMemcpy(this->B->host, this->A->host, (customWidth > -1 ? (this->A->height * customWidth) : this->A->length) * sizeof(float), cudaMemcpyHostToHost);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
	return true;
}

template<>
bool CopyTo<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	cudaError_t err;
	int N = (customWidth > -1 ? (customWidth * this->A->length) : this->A->length);
	for (int i = 0; i < this->A->batchSize; i++) {
		err = cudaMemcpy(this->B->host[i], this->A->host[i], N * sizeof(float), cudaMemcpyHostToHost);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
	return true;
}

__global__
void kernelConstantFill(int N, float c, float* A) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		A[i] = c;
	}
}

template<>
bool ConstantFill<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = this->A->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelConstantFill << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (N, this->c, queue->hostDevices[id][0][0]);
	return true;
}

__global__
void kernelConstantFillBatched(int N, float c, float** A) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;

	if (n < N) {
		A[batch][n] = c;
	}
}

template<>
bool ConstantFill<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = this->A->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, A->batchSize);
	kernelConstantFillBatched <<< blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (N, c, queue->devices[id][0]);
	return true;
}

__global__
void kernelScale(int N, float c, float* A) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		A[i] *= c;
	}
}

template<>
bool Scale<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = this->A->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelScale << < numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (N, this->c, queue->hostDevices[id][0][0]);
	return true;
}

__global__
void kernelScaleBatched(int N, float c, float** A) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;

	if (n < N) {
		A[batch][n] *= c;
	}
}

template<>
bool Scale<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = this->A->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, A->batchSize);
	kernelScaleBatched <<< blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (N, c, queue->devices[id][0]);
	return true;
}

__global__
void kernelSqrt(int N, float* A, float* B) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		B[i] = sqrt(A[i]);
	}
}

template<>
bool Sqrt<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = this->A->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelSqrt <<< numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (N, queue->hostDevices[id][0][0], queue->hostDevices[id][1][0]);
	return true;
}

__global__
void kernelSqrtBatched(int N, float** A, float** B) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		B[batch][i] = sqrt(A[batch][i]);
	}
}

template<>
bool Sqrt<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = this->A->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, A->batchSize);
	kernelSqrtBatched <<< blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (N, queue->devices[id][0], queue->devices[id][1]);
	return true;
}

__global__
void kernelTranspose(int N, int height, float* A, float* B) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		int row = i % height;
		int col = i / height;
		B[row * height + col] = A[i];
	}
}

template<>
bool Transpose<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = this->A->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelTranspose <<< numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (N, this->A->height, queue->hostDevices[id][0][0], queue->hostDevices[id][1][0]);
	return true;
}

__global__
void kernelTranposeBatched(int N, int height, float** A, float** B) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	if (i < N) {
		int row = i % height;
		int col = i / height;
		B[batch][row * height + col] = A[batch][i];
	}
}

template<>
bool Transpose<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = this->A->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, A->batchSize);
	kernelTranposeBatched <<< blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (N, this->A->height, queue->devices[id][0], queue->devices[id][1]);
	return true;
}

__global__
void kernelCondense(int N, int batchSize, float** A, float* B) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float sum = 0;
		for (int b = 0; b < batchSize; b++) {
			sum += A[b][i];
		}
		B[i] = sum;
	}
}

bool Condense::operate(OperationQueue* queue, int threadID) {
	int N = this->A->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelCondense <<< numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (N, this->A->batchSize, queue->devices[id][0], queue->hostDevices[id][1][0]);
	return true;
}

__global__
void kernelAdd(int N, float* A, float* B, float* C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = A[i] + B[i];
	}
}

template<>
bool Add<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = customWidth == -1 ? min(this->A->length, this->B->length) : (customWidth * this->A->height);
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelAdd <<< numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (N, queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][2][0]);
	return true;
}

__global__
void kernelAddBatched(float** A, float** B, float** C, int N) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;

	if (n < N) {
		C[batch][n] = A[batch][n] + B[batch][n];
	}
}

template<>
bool Add<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = customWidth == -1? min(this->A->length, this->B->length) : (customWidth * this->A->height);
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, A->batchSize);
	kernelAddBatched <<< blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (queue->devices[id][0], queue->devices[id][1], queue->devices[id][2], N);
	return true;
}

__global__
void kernelMultiply(int N, float* A, float* B, float* C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = A[i] * B[i];
	}
}

template<>
bool ElementMultiply<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = min(this->A->length, this->B->length);
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelMultiply <<< numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (N, queue->hostDevices[id][0][0], queue->hostDevices[id][1][0], queue->hostDevices[id][2][0]);
	return true;
}

__global__
void kernelMultiplyBatched(float** A, float** B, float** C, int N) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;

	if (n < N) {
		C[batch][n] = A[batch][n] * B[batch][n];
	}
}

template<>
bool ElementMultiply<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = min(this->A->length, this->B->length);
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, A->batchSize);
	kernelMultiplyBatched <<< blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (queue->devices[id][0], queue->devices[id][1], queue->devices[id][2], N);
	return true;
}

__global__
void kernelLinearCombo(int N, float c1, float* A, float c2, float* B, float* C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = c1 * A[i] + c2 * B[i];
	}
}

template<>
bool LinearCombo<Matrix>::operate(OperationQueue* queue, int threadID) {
	int N = A->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelLinearCombo <<< numBlocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >> > (N, c1, queue->hostDevices[id][0][0], c2, queue->hostDevices[id][1][0], queue->hostDevices[id][2][0]);
	return true;
}

__global__
void kernelLinearComboBatched(float c1, float** A, float c2, float** B, float** C, int N) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;

	if (n < N) {
		C[batch][n] = c1 * A[batch][n] + c2 * B[batch][n];
	}
}

template<>
bool LinearCombo<MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int N = A->length;
	int id = this->threadID.load();
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	dim3 blocks(numBlocks, A->batchSize);
	kernelLinearComboBatched <<< blocks, Utils::THREADS_PER_BLOCK, 0, queue->streams[id] >>> (c1, queue->devices[id][0], c2, queue->devices[id][1], queue->devices[id][2], N);
	return true;
}

template<>
bool MultiplyABC<Matrix, Matrix, Matrix>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cublasStatus_t stat = cublasSgemm(queue->handles[id], CUBLAS_OP_N, CUBLAS_OP_N, this->A->height, this->B->isLayerOutput? (this->B->width + 1):this->B->width, this->A->isLayerOutput? (this->A->width + 1):this->A->width, &ALPHA, queue->hostDevices[id][0][0], this->A->height, queue->hostDevices[id][1][0], this->B->height, &(overwrite ? BETA0 : BETA1), queue->hostDevices[id][2][0], this->C->height);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	return true;
}

template<>
bool MultiplyABC<MatrixBatch, Matrix, MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cublasStatus_t stat = cublasSgemmBatched(queue->handles[id], CUBLAS_OP_N, CUBLAS_OP_N, this->A->height, this->B->isLayerOutput? (this->B->width + 1):this->B->width, this->A->isLayerOutput? (this->A->width + 1):this->A->width, &ALPHA, queue->devices[id][0], this->A->height, queue->devices[id][1], this->B->height, &(overwrite ? BETA0 : BETA1), queue->devices[id][2], this->C->height, this->A->batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	return true;
}

template<>
bool MultiplyABC<Matrix, MatrixBatch, MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cublasStatus_t stat = cublasSgemmBatched(queue->handles[id], CUBLAS_OP_N, CUBLAS_OP_N, this->A->height, this->B->isLayerOutput? (this->B->width + 1):this->B->width, this->A->isLayerOutput? (this->A->width + 1):this->A->width, &ALPHA, queue->devices[id][0], this->A->height, queue->devices[id][1], this->B->height, &(overwrite ? BETA0 : BETA1), queue->devices[id][2], this->C->height, this->B->batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	return true;
}

template<>
bool MultiplyABC<MatrixBatch, MatrixBatch, MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cublasStatus_t stat = cublasSgemmBatched(queue->handles[id], CUBLAS_OP_N, CUBLAS_OP_N, this->A->height, this->B->isLayerOutput? (this->B->width + 1):this->B->width, this->A->isLayerOutput? (this->A->width + 1):this->A->width, &ALPHA, queue->devices[id][0], this->A->height, queue->devices[id][1], this->B->height, &(overwrite ? BETA0 : BETA1), queue->devices[id][2], this->C->height, this->A->batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	return true;
}

template<>
bool MultiplyAtBC<Matrix, Matrix, Matrix>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cublasStatus_t stat = cublasSgemm(Utils::HANDLE, CUBLAS_OP_T, CUBLAS_OP_N, this->A->isLayerOutput? (this->A->width + 1):this->A->width, this->B->isLayerOutput? (this->B->width + 1):this->B->width, this->A->height, &ALPHA, queue->hostDevices[id][0][0], this->A->height, queue->hostDevices[id][1][0], this->B->height, &(overwrite ? BETA0 : BETA1), queue->hostDevices[id][2][0], this->C->height);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	return true;
}

template<>
bool MultiplyAtBC<MatrixBatch, Matrix, MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cublasStatus_t stat = cublasSgemmBatched(Utils::HANDLE, CUBLAS_OP_T, CUBLAS_OP_N, this->A->isLayerOutput? (this->A->width + 1):this->A->width, this->B->isLayerOutput? (this->B->width + 1):this->B->width, this->A->height, &ALPHA, queue->devices[id][0], this->A->height, queue->devices[id][1], this->B->height, &(overwrite ? BETA0 : BETA1), queue->devices[id][2], this->C->height, this->A->batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	return true;
}

template<>
bool MultiplyAtBC<Matrix, MatrixBatch, MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cublasStatus_t stat = cublasSgemmBatched(Utils::HANDLE, CUBLAS_OP_T, CUBLAS_OP_N, this->A->isLayerOutput? (this->A->width + 1):this->A->width, this->B->isLayerOutput? (this->B->width + 1):this->B->width, this->A->height, &ALPHA, queue->devices[id][0], this->A->height, queue->devices[id][1], this->B->height, &(overwrite ? BETA0 : BETA1), queue->devices[id][2], this->C->height, this->B->batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	return true;
}

template<>
bool MultiplyAtBC<MatrixBatch, MatrixBatch, MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cublasStatus_t stat = cublasSgemmBatched(Utils::HANDLE, CUBLAS_OP_T, CUBLAS_OP_N, this->A->isLayerOutput? (this->A->width + 1):this->A->width, this->B->isLayerOutput? (this->B->width + 1):this->B->width, this->A->height, &ALPHA, queue->devices[id][0], this->A->height, queue->devices[id][1], this->B->height, &(overwrite ? BETA0 : BETA1), queue->devices[id][2], this->C->height, this->A->batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	return true;
}

template<>
bool MultiplyAtBtC<Matrix, Matrix, Matrix>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cublasStatus_t stat = cublasSgemm(Utils::HANDLE, CUBLAS_OP_T, CUBLAS_OP_T, this->A->isLayerOutput? (this->A->width + 1):this->A->width, this->B->height, this->A->height, &ALPHA, queue->hostDevices[id][0][0], this->A->height, queue->hostDevices[id][1][0], this->B->height, &(overwrite ? BETA0 : BETA1), queue->hostDevices[id][2][0], this->C->height);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	return true;
}

template<>
bool MultiplyAtBtC<MatrixBatch, Matrix, MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cublasStatus_t stat = cublasSgemmBatched(Utils::HANDLE, CUBLAS_OP_T, CUBLAS_OP_T, this->A->isLayerOutput? (this->A->width + 1):this->A->width, this->B->height, this->A->height, &ALPHA, queue->devices[id][0], this->A->height, queue->devices[id][1], this->B->height, &(overwrite ? BETA0 : BETA1), queue->devices[id][2], this->C->height, this->A->batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	return true;
}

template<>
bool MultiplyAtBtC<Matrix, MatrixBatch, MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cublasStatus_t stat = cublasSgemmBatched(Utils::HANDLE, CUBLAS_OP_T, CUBLAS_OP_T, this->A->isLayerOutput? (this->A->width + 1):this->A->width, this->B->height, this->A->height, &ALPHA, queue->devices[id][0], this->A->height, queue->devices[id][1], this->B->height, &(overwrite ? BETA0 : BETA1), queue->devices[id][2], this->C->height, this->B->batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	return true;
}

template<>
bool MultiplyAtBtC<MatrixBatch, MatrixBatch, MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cublasStatus_t stat = cublasSgemmBatched(Utils::HANDLE, CUBLAS_OP_T, CUBLAS_OP_T, this->A->isLayerOutput? (this->A->width + 1):this->A->width, this->B->height, this->A->height, &ALPHA, queue->devices[id][0], this->A->height, queue->devices[id][1], this->B->height, &(overwrite ? BETA0 : BETA1), queue->devices[id][2], this->C->height, this->A->batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	return true;
}

template<>
bool MultiplyABtC<Matrix, Matrix, Matrix>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cublasStatus_t stat = cublasSgemm(Utils::HANDLE, CUBLAS_OP_N, CUBLAS_OP_T, this->A->height, this->B->height, this->A->isLayerOutput? (this->A->width + 1):this->A->width, &ALPHA, queue->hostDevices[id][0][0], this->A->height, queue->hostDevices[id][1][0], this->B->height, &(overwrite ? BETA0 : BETA1), queue->hostDevices[id][2][0], this->C->height);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	return true;
}

template<>
bool MultiplyABtC<MatrixBatch, Matrix, MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cublasStatus_t stat = cublasSgemmBatched(Utils::HANDLE, CUBLAS_OP_N, CUBLAS_OP_T, this->A->height, this->B->height, this->A->isLayerOutput? (this->A->width + 1):this->A->width, &ALPHA, queue->devices[id][0], this->A->height, queue->devices[id][1], this->B->height, &(overwrite ? BETA0 : BETA1), queue->devices[id][2], this->C->height, this->A->batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	return true;
}

template<>
bool MultiplyABtC<Matrix, MatrixBatch, MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cublasStatus_t stat = cublasSgemmBatched(Utils::HANDLE, CUBLAS_OP_N, CUBLAS_OP_T, this->A->height, this->B->height, this->A->isLayerOutput? (this->A->width + 1):this->A->width, &ALPHA, queue->devices[id][0], this->A->height, queue->devices[id][1], this->B->height, &(overwrite ? BETA0 : BETA1), queue->devices[id][2], this->C->height, this->B->batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	return true;
}

template<>
bool MultiplyABtC<MatrixBatch, MatrixBatch, MatrixBatch>::operate(OperationQueue* queue, int threadID) {
	int id = this->threadID.load();
	cublasStatus_t stat = cublasSgemmBatched(Utils::HANDLE, CUBLAS_OP_N, CUBLAS_OP_T, this->A->height, this->B->height, this->A->isLayerOutput? (this->A->width + 1):this->A->width, &ALPHA, queue->devices[id][0], this->A->height, queue->devices[id][1], this->B->height, &(overwrite ? BETA0 : BETA1), queue->devices[id][2], this->C->height, this->A->batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
	return true;
}