#include "MatrixBatchOperations.h"

template<typename A>
void extendArray(A*& array, int oldLength, int newLength) {
	A* newArray;
	cudaError_t err = cudaMallocHost(&newArray, newLength * sizeof(A));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	if (array != NULL) {
		err = cudaMemcpy(newArray, array, oldLength * sizeof(A), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
		err = cudaFreeHost(array);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory deallocation failed: ") + cudaGetErrorString(err));
		}
	}
	array = newArray;
}

BBTrinary::BBTrinary(MatrixBatch& A, MatrixBatch& B, MatrixBatch& C) {
	this->A = &A;
	this->B = &B;
	this->C = &C;
}

void BBTrinary::applyToStream(StreamEnvironment stream) {
	if (stream.numDevices < 3) {
		extendArray(stream.deviceBatchSizes, stream.numDevices, 3);
		extendArray(stream.deviceLengths, stream.numDevices, 3);
	}
	stream.deviceBatchSizes[0] = max(stream.deviceBatchSizes[0], A->batchSize);
	stream.deviceBatchSizes[1] = max(stream.deviceBatchSizes[1], A->batchSize);
	stream.deviceBatchSizes[2] = max(stream.deviceBatchSizes[2], A->batchSize);
	stream.deviceLengths[0] = max(stream.deviceLengths[0], A->maxLength);
	stream.deviceLengths[1] = max(stream.deviceLengths[1], B->maxLength);
	stream.deviceLengths[2] = max(stream.deviceLengths[2], C->maxLength);
}

void BBTrinary::copyToDevice(StreamEnvironment stream) {
	cudaError_t err;
	for (int i = 0; i < A->batchSize; i++) {
		err = cudaMemcpyAsync(stream.hostDevices[0][i], A->host[i], A->length * sizeof(float), cudaMemcpyHostToDevice, stream.stream);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
	for (int i = 0; i < A->batchSize; i++) {
		err = cudaMemcpyAsync(stream.hostDevices[1][i], B->host[i], B->length * sizeof(float), cudaMemcpyHostToDevice, stream.stream);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
	for (int i = 0; i < A->batchSize; i++) {
		err = cudaMemcpyAsync(stream.hostDevices[2][i], C->host[i], C->length * sizeof(float), cudaMemcpyHostToDevice, stream.stream);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
}

void BBTrinary::copyToHost(StreamEnvironment stream) {
	cudaError_t err;
	for (int i = 0; i < A->batchSize; i++) {
		err = cudaMemcpyAsync(C->host[i], stream.hostDevices[2][i], outputLength * sizeof(float), cudaMemcpyDeviceToHost, stream.stream);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}

	}
}

MBTrinary::MBTrinary(Matrix2& A, MatrixBatch& B, MatrixBatch& C) {
	this->A = &A;
	this->B = &B;
	this->C = &C;
}

void MBTrinary::applyToStream(StreamEnvironment stream) {
	if (stream.numDevices < 3) {
		extendArray(stream.deviceBatchSizes, stream.numDevices, 3);
		extendArray(stream.deviceLengths, stream.numDevices, 3);
	}
	stream.deviceBatchSizes[0] = max(stream.deviceBatchSizes[0], B->batchSize);
	stream.deviceBatchSizes[1] = max(stream.deviceBatchSizes[1], B->batchSize);
	stream.deviceBatchSizes[2] = max(stream.deviceBatchSizes[2], B->batchSize);
	stream.deviceLengths[0] = max(stream.deviceLengths[0], A->maxLength);
	stream.deviceLengths[1] = max(stream.deviceLengths[1], B->maxLength);
	stream.deviceLengths[2] = max(stream.deviceLengths[2], C->maxLength);
}

void MBTrinary::copyToDevice(StreamEnvironment stream) {
	cudaError_t err;
	for (int i = 0; i < B->batchSize; i++) {
		err = cudaMemcpyAsync(stream.hostDevices[0][i], A->host, A->length * sizeof(float), cudaMemcpyHostToDevice, stream.stream);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
	for (int i = 0; i < B->batchSize; i++) {
		err = cudaMemcpyAsync(stream.hostDevices[1][i], B->host[i], B->length * sizeof(float), cudaMemcpyHostToDevice, stream.stream);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
	for (int i = 0; i < B->batchSize; i++) {
		err = cudaMemcpyAsync(stream.hostDevices[2][i], C->host[i], C->length * sizeof(float), cudaMemcpyHostToDevice, stream.stream);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
}

void MBTrinary::copyToHost(StreamEnvironment stream) {
	cudaError_t err;
	for (int i = 0; i < B->batchSize; i++) {
		err = cudaMemcpyAsync(C->host[i], stream.hostDevices[2][i], outputLength * sizeof(float), cudaMemcpyDeviceToHost, stream.stream);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}

	}
}

__global__
void kernelAddOperationBatch(float** A, float** B, float** C, int N) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;

	if (n < N) {
		C[batch][n] = A[batch][n] + B[batch][n];
	}
}

void BBAdd::operate(StreamEnvironment stream) {
	int N = A->length;
	outputLength = N;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, A->batchSize);
	kernelAddOperationBatch <<< blocks, Utils::THREADS_PER_BLOCK, 0, stream.stream>> > (stream.devices[0], stream.devices[1], stream.devices[2], N);
}

__global__
void kernelMultiplyOperationBatch(float** A, float** B, float** C, int N) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;

	if (n < N) {
		C[batch][n] = A[batch][n] * B[batch][n];
	}
}

void BBElementMultiply::operate(StreamEnvironment stream) {
	int N = A->length;
	outputLength = N;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, A->batchSize);
	kernelMultiplyOperationBatch << < blocks, Utils::THREADS_PER_BLOCK, 0, stream.stream >> > (stream.devices[0], stream.devices[1], stream.devices[2], N);
}

__global__
void kernelLinearComboOperationBatch(float c1, float** A, float c2, float** B, float** C, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y;
	if (i < N) {
		C[b][i] = c1 * A[b][i] + c2 * B[b][i];
	}
}

void BBLinearCombo::operate(StreamEnvironment stream) {
	int N = A->length;
	outputLength = N;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1);
	dim3 blocks(numBlocks, A->batchSize);
	kernelLinearComboOperationBatch <<< blocks, Utils::THREADS_PER_BLOCK, 0, stream.stream >> > (c1, stream.devices[0], c2, stream.devices[1], stream.devices[2], N);
}

void BBMultiplyABC::operate(StreamEnvironment stream) {
	outputLength = A->height * B->width;
	cublasStatus_t stat = cublasSgemmBatched(stream.handle, CUBLAS_OP_N, CUBLAS_OP_N, A->height, B->width, A->width, &ALPHA, stream.devices[0], A->height, stream.devices[1], B->height, &(overwrite ? BETA0 : BETA1), stream.devices[2], C->height, C->batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
}

void BBMultiplyAtBC::operate(StreamEnvironment stream) {
	outputLength = A->width * B->width;
	cublasStatus_t stat = cublasSgemmBatched(stream.handle, CUBLAS_OP_T, CUBLAS_OP_N, A->width, B->width, A->height, &ALPHA, stream.devices[0], A->height, stream.devices[1], B->height, &(overwrite ? BETA0 : BETA1), stream.devices[2], C->height, C->batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
}

void BBMultiplyAtBtC::operate(StreamEnvironment stream) {
	outputLength = A->width * B->height;
	cublasStatus_t stat = cublasSgemmBatched(stream.handle, CUBLAS_OP_T, CUBLAS_OP_T, A->width, B->height, A->height, &ALPHA, stream.devices[0], A->height, stream.devices[1], B->height, &(overwrite ? BETA0 : BETA1), stream.devices[2], C->height, C->batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
}

void BBMultiplyABtC::operate(StreamEnvironment stream) {
	outputLength = A->height * B->height;
	cublasStatus_t stat = cublasSgemmBatched(stream.handle, CUBLAS_OP_N, CUBLAS_OP_T, A->height, B->height, A->width, &ALPHA, stream.devices[0], A->height, stream.devices[1], B->height, &(overwrite ? BETA0 : BETA1), stream.devices[2], C->height, C->batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
}