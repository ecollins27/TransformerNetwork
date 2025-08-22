#include "OperationQueue.h"

OperationQueue::OperationQueue(int numStreams) {
	this->numThreads = numStreams;
	this->numDevices = 0;
	this->deviceBatchSizes = NULL;
	this->deviceLengths = NULL;
	this->devices = NULL;
	this->hostDevices = NULL;
	threads = new thread[numThreads];
	streams = new cudaStream_t[numThreads];
	handles = new cublasHandle_t[numThreads];
	deviceLocks = new atomic<bool>[numThreads];
	devicesUsed.store(0);
	for (int i = 0; i < numThreads; i++) {
		deviceLocks[i].store(true);
		cudaStreamCreate(&streams[i]);
		cublasCreate(&handles[i]);
		cublasSetStream(handles[i], streams[i]);
	}
}

int OperationQueue::getMinIndex(vector<Operation*> v) {
	if (v.size() == 0) {
		return -1;
	}
	bool refFalse;
	int prereqs;
	bool allAllocated = true;
	for (int i = 0; i < min(maxIndex.load() + 8, (int)operations.size()); i++) {
		refFalse = false;
		prereqs = v[i]->getPrereqsUnmet(this);
		if (v[i]->completed.load() == 1 && prereqs == 0 && v[i]->operationAllocated.compare_exchange_strong(refFalse, true)) {
			//printf("Operation Allocated: %d %d\n", i, v[i]->operationAllocated.load());
			return i;
		}
		else if (!v[i]->operationAllocated.load()) {
			allAllocated = false;
		}
	}
	return allAllocated? -2:-1;
}

bool OperationQueue::operationsAllocated(vector<Operation*> operations) {
	for (int i = 0; i < operations.size(); i++) {
		if (operations[i]->completed.load() > 0 || !operations[i]->operationAllocated.load()) {
			return false;
		}
	}
	return true;
}

int numUnallocated(vector<Operation*> operations) {
	int n = 0;
	for (int i = 0; i < operations.size(); i++) {
		n += operations[i]->operationAllocated.load() ? 0 : 1;
	}
	return n;
}

void OperationQueue::threadRun(int threadID) {
	Operation* operation = NULL;
	int index = -1;
	int size = operations.size();
	bool success;
	while (!operationsAllocated(operations)) {
		index = -1;
		while (index == -1) {
			index = getMinIndex(operations);
			//if (threadID == 0) {
			//	printf("\rNum Threads Taken: %d", devicesUsed.load());
			//}
		}
		//if (threadID == 0) {
		//	printf("\n");
		//}
		if (index == -2) {
			continue;
		}
		operation = operations[index];
		printf("Thread %d performing operation %s %d\n", threadID, typeid(*operation).name(), index);
		success = operation->operate(this, threadID);
		maxIndex.store(max(maxIndex.load(), index));
		if (success) {
			operation->completed.store(0);
		}
		else {
			operation->operationAllocated.store(false);
		}
	}
}

void OperationQueue::run() {
	for (int i = 0; i < numThreads; i++) {
		threads[i] = thread(&OperationQueue::threadRun, this, i);
	}
	for (int i = 0; i < numThreads; i++) {
		threads[i].join();
	}
}

void OperationQueue::reset() {
	maxIndex.store(0);
	for (int i = 0; i < operations.size(); i++) {
		//printf("%d: %p  %s\n", i, operations[i], typeid(*operations[i]).name());
		operations[i]->completed.store(1);
		operations[i]->operationAllocated.store(false);
	}
}

void OperationQueue::enqueue(Operation* operation) {
	if (dynamic_cast<DOperation*>(operation) != NULL) {
		((DOperation*) operation)->addDeviceCopies(operations);
		operations.emplace_back(operation);
		((DOperation*) operation)->addHostCopies(operations);
		((DOperation*)operation)->threadID.store(-2);
	}
	else {
		operations.emplace_back(operation);
	}
}

void OperationQueue::finalize() {
	for (int i = 0; i < operations.size(); i++) {
		operations[i]->findPrereqs(operations, i);
		operations[i]->applyToStream(this);
	}
	for (int i = 0; i < operations.size(); i++) {
		printf("%s: %d\n", typeid(*(operations[i])).name(), operations[i]->prereqDepth);
	}
	printf("\n");
	allocateDeviceMemory();
	this->reset();
}

void OperationQueue::allocateDeviceMemory() {
	cudaError_t err = cudaMallocHost(&devices, numThreads * sizeof(float***));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	err = cudaMallocHost(&hostDevices, numThreads * sizeof(float***));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	for (int i = 0; i < numThreads; i++) {
		err = cudaMallocHost(&devices[i], numDevices * sizeof(float**));
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
		}
		err = cudaMallocHost(&hostDevices[i], numDevices * sizeof(float**));
		if (err != cudaSuccess) {
			throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
		}
		for (int j = 0; j < numDevices; j++) {
			err = cudaMalloc(&devices[i][j], deviceBatchSizes[j] * sizeof(float*));
			if (err != cudaSuccess) {
				throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
			}
			err = cudaMallocHost(&hostDevices[i][j], deviceBatchSizes[j] * sizeof(float*));
			if (err != cudaSuccess) {
				throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
			}
			for (int k = 0; k < deviceBatchSizes[j]; k++) {
				err = cudaMalloc(&hostDevices[i][j][k], deviceLengths[j] * sizeof(float));
				if (err != cudaSuccess) {
					throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
				}
			}
			err = cudaMemcpy(devices[i][j], hostDevices[i][j], deviceBatchSizes[j] * sizeof(float*), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
			}
		}
	}
}

long long OperationQueue::getDeviceMemory() {
	long long sum = 0;
	for (int i = 0; i < numDevices; i++) {
		sum += deviceBatchSizes[i] * (deviceLengths[i] + 1);
	}
	return sum * numThreads;
}

int OperationQueue::getNextAvailableThread() {
	for (int i = 0; i < numThreads; i++) {
		if (deviceLocks[i].load()) {
			return i;
		}
	}
	return -1;
}