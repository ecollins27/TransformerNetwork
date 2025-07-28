#include "PropagationQueue.h"

PropagationQueue::PropagationQueue(int numStreams) {
	this->numThreads = numStreams;
	this->numDevices = 0;
	this->deviceBatchSizes = NULL;
	this->deviceLengths = NULL;
	this->devices = NULL;
	this->hostDevices = NULL;
	threads = new thread[numThreads];
	deviceLocks = new atomic<bool>[numThreads];
	for (int i = 0; i < numThreads; i++) {
		deviceLocks[i].store(true);
	}
}

void PropagationQueue::threadRun(int threadID) {
	Operation* operation;
	while (!operationQueue.empty()) {
		while (!queueLock.try_lock()) {
			operation = operationQueue.top();
			operationQueue.pop();
			queueLock.unlock();
		}
		operation->operate(this, threadID);
		operation->completed = 1;
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
	while (!operationQueue.empty()) {
		operationQueue.pop();
	}
	for (int i = 0; i < operations.size(); i++) {
		operations[i]->completed = 0;
		operationQueue.push(operations[i]);
	}
}

void PropagationQueue::enqueueOperation(Operation* operation) {
	if (dynamic_cast<GPUOperation*>(operation) != NULL) {
		((GPUOperation*) operation)->addDeviceCopies(operations);
		operations.emplace_back(operation);
		((GPUOperation*) operation)->addHostCopies(operations);
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
		err = cudaMallocHost(&hostDevices, numDevices * sizeof(float**));
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
				throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
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
	int thread = 0;
	while (!deviceLocks[thread].load()) {
		thread++;
		if (thread >= numThreads) {
			thread = 0;
		}
	}
	return thread;
}