#include "Operation.h"
#include "PropagationQueue.h"

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
void HostToDeviceCopy<Matrix2>::operate(PropagationQueue* queue, int threadID) {
	if (!this->idFound->load()) {
		this->idFound->store(true);
		*copyID = queue->getNextAvailableThread();
		queue->deviceLocks[*copyID].store(false);
	}
	cudaError_t err = cudaMemcpy(queue->hostDevices[*copyID][this->deviceNum][0], A->host, A->length * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
}

template<>
void HostToDeviceCopy<MatrixBatch>::operate(PropagationQueue* queue, int threadID) {
	if (!this->idFound->load()) {
		this->idFound->store(true);
		*copyID = queue->getNextAvailableThread();
		queue->deviceLocks[*copyID].store(false);
	}
	cudaError_t err;
	for (int i = 0; i < A->batchSize; i++) {
		err = cudaMemcpy(queue->hostDevices[*copyID][this->deviceNum][i], A->host[i], A->length * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
}

template<typename Type>
void HostToDeviceCopy<Type>::operate(PropagationQueue* queue, int threadID) {
	throw runtime_error("All passed parameters must be instance of Matrix or MatrixBatch");
}

template<>
void HostToDeviceCopy<Matrix2>::applyToStream(PropagationQueue* queue) {
	if (queue->numDevices <= deviceNum) {
		extendArray(queue->deviceBatchSizes, queue->numDevices, deviceNum + 1);
		extendArray(queue->deviceLengths, queue->numDevices, deviceNum + 1);
		for (int i = queue->numDevices; i <= deviceNum; i++) {
			queue->deviceBatchSizes[i] = 0;
			queue->deviceLengths[i] = 0;
		}
	}
	queue->deviceBatchSizes[deviceNum] = max(queue->deviceBatchSizes[deviceNum], 1);
	queue->deviceLengths[deviceNum] = max(queue->deviceLengths[deviceNum], A->maxLength);
}

template<>
void HostToDeviceCopy<MatrixBatch>::applyToStream(PropagationQueue* queue) {
	if (queue->numDevices <= deviceNum) {
		extendArray(queue->deviceBatchSizes, queue->numDevices, deviceNum + 1);
		extendArray(queue->deviceLengths, queue->numDevices, deviceNum + 1);
		for (int i = queue->numDevices; i <= deviceNum; i++) {
			queue->deviceBatchSizes[i] = 0;
			queue->deviceLengths[i] = 0;
		}
	}
	queue->deviceBatchSizes[deviceNum] = max(queue->deviceBatchSizes[deviceNum], A->batchSize);
	queue->deviceLengths[deviceNum] = max(queue->deviceLengths[deviceNum], A->maxLength);
}

template<typename Type>
void HostToDeviceCopy<Type>::applyToStream(PropagationQueue* queue) {
	throw runtime_error("All passed parameters must be instance of Matrix or MatrixBatch");
}

template<>
void DeviceToHostCopy<Matrix2>::operate(PropagationQueue* queue, int threadID) {
	if (this->idFound->load()) {
		this->idFound->store(false);
	}
	cudaError_t err = cudaMemcpy(A->host, queue->hostDevices[*copyID][this->deviceNum][0], A->length * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
	}
	if (!queue->deviceLocks[*copyID].load()) {
		queue->deviceLocks[*copyID].store(true);
	}
}

template<>
void DeviceToHostCopy<MatrixBatch>::operate(PropagationQueue* queue, int threadID) {
	if (this->idFound->load()) {
		this->idFound->store(false);
	}
	cudaError_t err;
	for (int i = 0; i < A->batchSize; i++) {
		err = cudaMemcpy(A->host[i], queue->hostDevices[*copyID][this->deviceNum][i], A->length * sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
		}
	}
	if (!queue->deviceLocks[*copyID].load()) {
		queue->deviceLocks[*copyID].store(true);
	}
}

template<typename Type>
void DeviceToHostCopy<Type>::operate(PropagationQueue* queue, int threadID) {
	throw runtime_error("All passed parameters must be instance of Matrix or MatrixBatch");
}

template<>
void DeviceToHostCopy<Matrix2>::applyToStream(PropagationQueue* queue) {
	if (queue->numDevices <= deviceNum) {
		extendArray(queue->deviceBatchSizes, queue->numDevices, deviceNum + 1);
		extendArray(queue->deviceLengths, queue->numDevices, deviceNum + 1);
		for (int i = queue->numDevices; i <= deviceNum; i++) {
			queue->deviceBatchSizes[i] = 0;
			queue->deviceLengths[i] = 0;
		}
	}
	queue->deviceBatchSizes[deviceNum] = max(queue->deviceBatchSizes[deviceNum], 1);
	queue->deviceLengths[deviceNum] = max(queue->deviceLengths[deviceNum], A->maxLength);
}

template<>
void DeviceToHostCopy<MatrixBatch>::applyToStream(PropagationQueue* queue) {
	if (queue->numDevices <= deviceNum) {
		extendArray(queue->deviceBatchSizes, queue->numDevices, deviceNum + 1);
		extendArray(queue->deviceLengths, queue->numDevices, deviceNum + 1);
		for (int i = queue->numDevices; i <= deviceNum; i++) {
			queue->deviceBatchSizes[i] = 0;
			queue->deviceLengths[i] = 0;
		}
	}
	queue->deviceBatchSizes[deviceNum] = max(queue->deviceBatchSizes[deviceNum], A->batchSize);
	queue->deviceLengths[deviceNum] = max(queue->deviceLengths[deviceNum], A->maxLength);
}

template<typename Type>
void DeviceToHostCopy<Type>::applyToStream(PropagationQueue* queue) {
	throw runtime_error("All passed parameters must be instance of Matrix or MatrixBatch");
}

void GPUOperation::applyToStream(PropagationQueue* queue) {
	return;
}

void GPUOperation::findPrereqs(vector<Operation*> operations, int index) {
	return;
}

template<typename TypeA>
Unary<TypeA>::Unary(TypeA& A) {
	this->A = &A;
	this->output = &A + 1;
	completed = false;
	prereq = NULL;
}

template<typename TypeA>
int Unary<TypeA>::getPrereqsUnmet() {
	return prereq->completed;
}

template<typename TypeA>
void Unary<TypeA>::addDeviceCopies(vector<Operation*>& operations) {
	Operation* ACopy = new HostToDeviceCopy(this, this->A, 0);
	operations.emplace_back(ACopy);
	prereq = ACopy;
}

template<typename TypeA>
void Unary<TypeA>::addHostCopies(vector<Operation*>& operations) {
	DeviceToHostCopy<TypeA>* ACopy = new DeviceToHostCopy(this, this->A, 0);
	operations.emplace_back(ACopy);
	ACopy->prereq = this;
}

template<typename TypeA, typename TypeB>
Binary<TypeA, TypeB>::Binary(TypeA& A, TypeB& B) {
	this->A = &A;
	this->B = &B;
	this->output = &A + 1;
	completed = false;
	prereqA = NULL;
}

template<typename TypeA, typename TypeB>
int Binary<TypeA, TypeB>::getPrereqsUnmet() {
	return prereqA->completed;
}

template<typename TypeA, typename TypeB>
void Binary<TypeA, TypeB>::addDeviceCopies(vector<Operation*>& operations) {
	Operation* ACopy = new HostToDeviceCopy(this, this->A, 0);
	operations.emplace_back(ACopy);
	this->prereqA = ACopy;
}

template<typename TypeA, typename TypeB>
void Binary<TypeA, TypeB>::addHostCopies(vector<Operation*>& operations) {
	DeviceToHostCopy<TypeB>* BCopy = new DeviceToHostCopy(this, this->B, 1);
	operations.emplace_back(BCopy);
	BCopy->prereq = this;
}

template<typename TypeA, typename TypeB, typename TypeC>
int Trinary<TypeA, TypeB, TypeC>::getPrereqsUnmet() {
	return (prereqA == NULL? 0 : prereqA->completed) + (prereqB == NULL? 0 : prereqB->completed);
}

template<>
void MultiplyABC<Matrix2, Matrix2, Matrix2>::operate(PropagationQueue* queue, int threadID) {
	cublasStatus_t stat = cublasSgemm(Utils::HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, this->A->height, this->B->width, this->A->width, &ALPHA, queue->hostDevices[this->threadID][0][0], this->A->height, queue->hostDevices[this->threadID][1][0], this->B->height, &(overwrite ? BETA0 : BETA1), queue->hostDevices[this->threadID][2][0], this->C->height);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(string("cuBLAS multiplication failed: ") + cublasGetStatusString(stat));
	}
}