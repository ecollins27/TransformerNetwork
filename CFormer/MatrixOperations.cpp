#include "MatrixOperations.h"

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

MUnary::MUnary(Matrix2& A) {
	this->A = &A;
	this->output = &A;
	ACopied = false;
}

void MUnary::applyToStream(StreamEnvironment stream) {
	if (stream.numDevices < 1) {
		extendArray(stream.deviceBatchSizes, stream.numDevices, 1);
		extendArray(stream.deviceLengths, stream.numDevices, 1);
		stream.deviceBatchSizes[0] = 0;
		stream.deviceLengths[0] = 0;
		stream.numDevices = 1;
	}
	stream.deviceBatchSizes[0] = max(stream.deviceBatchSizes[0], 1);
	stream.deviceLengths[0] = max(stream.deviceLengths[0], A->maxLength);
}

void MUnary::copyToDevice(StreamEnvironment stream, int index) {
	if (!ACopied && index >= APrereq) {
		cudaError_t err = cudaMemcpyAsync(stream.hostDevices[0][0], A->host, A->length * sizeof(float), cudaMemcpyHostToDevice, stream.stream);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
		ACopied = true;
	}
}

void MUnary::copyToHost(StreamEnvironment stream) {
	cudaError_t err = cudaMemcpyAsync(A->host, stream.hostDevices[0][0], A->length * sizeof(float), cudaMemcpyDeviceToHost, stream.stream);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
}

void MUnary::findPrereqs(vector<Operation*> operations) {
	APrereq = -1;
	for (int i = 0; i < operations.size(); i++) {
		if (operations[i]->output == A) {
			APrereq = i;
		}
	}
}

MBinary::MBinary(Matrix2& A, Matrix2& B) {
	this->A = &A;
	this->B = &B;
	this->output = &B;
	ACopied = false;
}

void MBinary::applyToStream(StreamEnvironment stream) {
	if (stream.numDevices < 2) {
		extendArray(stream.deviceBatchSizes, stream.numDevices, 2);
		extendArray(stream.deviceLengths, stream.numDevices, 2);
		for (int i = stream.numDevices; i < 2; i++) {
			stream.deviceBatchSizes[i] = 0;
			stream.deviceLengths[i] = 0;
		}
		stream.numDevices = 2;
	}
	stream.deviceBatchSizes[0] = max(stream.deviceBatchSizes[0], 1);
	stream.deviceBatchSizes[1] = max(stream.deviceBatchSizes[1], 1);
	stream.deviceLengths[0] = max(stream.deviceLengths[0], A->maxLength);
	stream.deviceLengths[1] = max(stream.deviceLengths[1], B->maxLength);
}

void MBinary::copyToDevice(StreamEnvironment stream, int index) {
	if (!ACopied && index >= APrereq) {
		cudaError_t err = cudaMemcpyAsync(stream.hostDevices[0][0], A->host, A->length * sizeof(float), cudaMemcpyHostToDevice, stream.stream);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
		ACopied = true;
	}
}

void MBinary::copyToHost(StreamEnvironment stream) {
	cudaError_t err = cudaMemcpyAsync(B->host, stream.hostDevices[1][0], B->maxLength * sizeof(float), cudaMemcpyDeviceToHost, stream.stream);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
}

void MBinary::findPrereqs(vector<Operation*> operations) {
	APrereq = -1;
	for (int i = 0; i < operations.size(); i++) {
		if (operations[i]->output == A) {
			APrereq = i;
		}
	}
}

MMTrinary::MMTrinary(Matrix2& A, Matrix2& B, Matrix2& C) {
	this->A = &A;
	this->B = &B;
	this->C = &C;
	this->output = &C;
	ACopied = false;
	BCopied = false;
	CCopied = false;
}

void MMTrinary::applyToStream(StreamEnvironment stream) {
	if (stream.numDevices < 3) {
		extendArray(stream.deviceBatchSizes, stream.numDevices, 3);
		extendArray(stream.deviceLengths, stream.numDevices, 3);
		for (int i = stream.numDevices; i < 3; i++) {
			stream.deviceBatchSizes[i] = 0;
			stream.deviceLengths[i] = 0;
		}
		stream.numDevices = 3;
	}
	stream.deviceBatchSizes[0] = max(stream.deviceBatchSizes[0], 1);
	stream.deviceBatchSizes[1] = max(stream.deviceBatchSizes[1], 1);
	stream.deviceBatchSizes[2] = max(stream.deviceBatchSizes[2], 1);
	stream.deviceLengths[0] = max(stream.deviceLengths[0], A->maxLength);
	stream.deviceLengths[1] = max(stream.deviceLengths[1], B->maxLength);
	stream.deviceLengths[2] = max(stream.deviceLengths[2], C->maxLength);
}

void MMTrinary::copyToDevice(StreamEnvironment stream, int index) {
	if (!ACopied && index >= APrereq) {
		cudaError_t err = cudaMemcpyAsync(stream.hostDevices[0][0], A->host, A->length * sizeof(float), cudaMemcpyHostToDevice, stream.stream);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
		ACopied = true;
	} if (!BCopied && index >= BPrereq) {
		cudaError_t err = cudaMemcpyAsync(stream.hostDevices[1][0], B->host, B->length * sizeof(float), cudaMemcpyHostToDevice, stream.stream);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
		BCopied = true;
	} if (!CCopied && index >= CPrereq) {
		cudaError_t err = cudaMemcpyAsync(stream.hostDevices[2][0], C->host, C->length * sizeof(float), cudaMemcpyHostToDevice, stream.stream);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
		CCopied = true;
	}
}

void MMTrinary::copyToHost(StreamEnvironment stream) {
	cudaError_t err = cudaMemcpyAsync(C->host, stream.hostDevices[2][0], outputLength * sizeof(float), cudaMemcpyDeviceToHost, stream.stream);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
}

void MMTrinary::findPrereqs(vector<Operation*> operations) {
	APrereq = -1;
	BPrereq = -1;
	CPrereq = -1;
	for (int i = 0; i < operations.size(); i++) {
		if (operations[i]->output == A) {
			APrereq = i;
		} if (operations[i]->output == B) {
			BPrereq = i;
		} if (operations[i]->output == C) {
			CPrereq = i;
		}
	}
}

__global__
void kernelAddOperation(int N, float* A, float* B, float* C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = A[i] + B[i];
	}
}

void MMAdd::operate(StreamEnvironment stream) {
	int N = A->length;
	outputLength = N;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelAddOperation <<< numBlocks, Utils::THREADS_PER_BLOCK, 0, stream.stream>>> (N, stream.hostDevices[0][0], stream.hostDevices[1][0], stream.hostDevices[2][0]);
}

__global__
void kernelMultiplyOperation(int N, float* A, float* B, float* C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = A[i] * B[i];
	}
}

void MMElementMultiply::operate(StreamEnvironment stream) {
	int N = A->length;
	outputLength = N;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelMultiplyOperation <<< numBlocks, Utils::THREADS_PER_BLOCK, 0, stream.stream >> > (N, stream.hostDevices[0][0], stream.hostDevices[1][0], stream.hostDevices[2][0]);
}

__global__
void kernelLinearComboOperation(float c1, float* A, float c2, float* B, float* C, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = c1 * A[i] + c2 * B[i];
	}
}

void MMLinearCombo::operate(StreamEnvironment stream) {
	int N = A->length;
	outputLength = N;
	int numBlocks = (N + Utils::THREADS_PER_BLOCK - 1) / Utils::THREADS_PER_BLOCK;
	kernelLinearComboOperation << < numBlocks, Utils::THREADS_PER_BLOCK, 0, stream.stream >> > (c1, stream.hostDevices[0][0], c2, stream.hostDevices[1][0], stream.hostDevices[2][0], N);
}

void MMMultiplyABC::operate(StreamEnvironment stream) {
	outputLength = A->height * B->width;
	cublasStatus_t stat = cublasSgemm(stream.handle, CUBLAS_OP_N, CUBLAS_OP_N, A->height, B->width, A->width, &ALPHA, stream.hostDevices[0][0], A->height, stream.hostDevices[1][0], B->height, &(overwrite ? BETA0 : BETA1), stream.hostDevices[2][0], C->height);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
}

void MMMultiplyAtBC::operate(StreamEnvironment stream) {
	outputLength = A->width * B->width;
	cublasStatus_t stat = cublasSgemm(stream.handle, CUBLAS_OP_T, CUBLAS_OP_N, A->width, B->width, A->height, &ALPHA, stream.hostDevices[0][0], A->height, stream.hostDevices[1][0], B->height, &(overwrite ? BETA0 : BETA1), stream.hostDevices[2][0], C->height);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
}

void MMMultiplyAtBtC::operate(StreamEnvironment stream) {
	outputLength = A->width * B->height;
	cublasStatus_t stat = cublasSgemm(stream.handle, CUBLAS_OP_T, CUBLAS_OP_T, A->width, B->height, A->height, &ALPHA, stream.hostDevices[0][0], A->height, stream.hostDevices[1][0], B->height, &(overwrite ? BETA0 : BETA1), stream.hostDevices[2][0], C->height);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
}

void MMMultiplyABtC::operate(StreamEnvironment stream) {
	outputLength = A->height * B->height;
	cublasStatus_t stat = cublasSgemm(stream.handle, CUBLAS_OP_N, CUBLAS_OP_T, A->height, B->height, A->width, &ALPHA, stream.hostDevices[0][0], A->height, stream.hostDevices[1][0], B->height, &(overwrite ? BETA0 : BETA1), stream.hostDevices[2][0], C->height);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
}