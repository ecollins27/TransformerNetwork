#include "PropagationQueue.h"

PropagationQueue::PropagationQueue(int numStreams) {
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

int PropagationQueue::getMinIndex(vector<Operation*> v) {
	if (v.size() == 0) {
		return -1;
	}
	bool refFalse;
	int prereqs;
	bool allAllocated = true;
	for (int i = 0; i < v.size(); i++) {
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

bool PropagationQueue::operationsAllocated(vector<Operation*> operations) {
	for (int i = 0; i < operations.size(); i++) {
		if (operations[i]->completed.load() > 0 || !operations[i]->operationAllocated.load()) {
			return false;
		}
	}
	return true;
}

void PropagationQueue::threadRun(int threadID) {
	Operation* operation = NULL;
	int index = -1;
	int size = operations.size();
	bool success;
	while (!operationsAllocated(operations)) {
		index = -1;
		while (index == -1) {
			index = getMinIndex(operations);
		}
		if (index == -2) {
			continue;
		}
		operation = operations[index];
		success = operation->operate(this, threadID);
		if (success) {
			operation->completed.store(0);
		}
		else {
			operation->operationAllocated.store(false);
		}
	}
}

void PropagationQueue::run() {
	for (int i = 0; i < numThreads; i++) {
		threads[i] = thread(&PropagationQueue::threadRun, this, i);
	}
	for (int i = 0; i < numThreads; i++) {
		threads[i].join();
	}
}

void PropagationQueue::reset() {
	for (int i = 0; i < operations.size(); i++) {
		operations[i]->completed.store(1);
		operations[i]->operationAllocated.store(false);
	}
}

void PropagationQueue::enqueueOperation(Operation* operation) {
	if (dynamic_cast<GPUOperation*>(operation) != NULL) {
		((GPUOperation*) operation)->addDeviceCopies(operations);
		operations.emplace_back(operation);
		((GPUOperation*) operation)->addHostCopies(operations);
		((GPUOperation*)operation)->threadID.store(-1);
	}
	else {
		operations.emplace_back(operation);
	}
}

void PropagationQueue::finalize() {
	for (int i = 0; i < operations.size(); i++) {
		operations[i]->findPrereqs(operations, i);
		operations[i]->applyToStream(this);
	}
	allocateDeviceMemory();
	for (int i = 0; i < operations.size(); i++) {
		operations[i]->completed.store(1);
		operations[i]->operationAllocated.store(false);
	}
	//topologicalSortOperations();
	//for (int i = 0; i < operations.size(); i++) {
	//	operations[i]->completed = 1;
	//}
}

void PropagationQueue::allocateDeviceMemory() {
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

long long PropagationQueue::getDeviceMemory() {
	long long sum = 0;
	for (int i = 0; i < numDevices; i++) {
		sum += deviceBatchSizes[i] * (deviceLengths[i] + 1);
	}
	return sum * numThreads;
}

int PropagationQueue::getNextAvailableThread() {
	for (int i = 0; i < numThreads; i++) {
		if (deviceLocks[i].load()) {
			return i;
		}
	}
	return -1;
}