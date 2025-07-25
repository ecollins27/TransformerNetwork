#include "PropagationQueue.h"

PropagationQueue::PropagationQueue(int numStreams) {
	this->numStreams = numStreams;
	streams = new StreamEnvironment[numStreams];
}

void PropagationQueue::start() {
	if (operations.size() <= 0) {
		throw runtime_error("PropagationQueue must be non-empty");
	}
	Operation* operation;
	for (int i = 0; i < min(numStreams, (int)operations.size()); i++) {
		operations[i]->copyToDevice(streams[i], -1);
	}
	for (int i = 0; i < operations.size(); i++) {
		operation = operations[i];
		operation->operate(streams[i % numStreams]);
		cudaStreamSynchronize(streams[i % numStreams].stream);
		operation->copyToHost(streams[i % numStreams]);
		for (int j = 1; j <= numStreams; j++) {
			operations[i + j]->copyToDevice(streams[(i + j) % numStreams], i);
		}
	}
}

void PropagationQueue::enqueueOperation(Operation* operation) {
	operation->applyToStream(streams[operations.size() % numStreams]);
	operation->findPrereqs(operations);
	operations.emplace_back(operation);
}

long long PropagationQueue::getDeviceMemory() {
	long long sum = 0;
	for (int i = 0; i < numStreams; i++) {
		sum += streams[i].getDeviceMemory();
	}
	return sum;
}

StreamEnvironment::StreamEnvironment() {
	cudaStreamCreate(&stream);
	cublasCreate(&handle);
	cublasSetStream(handle, stream);

	numDevices = 0;
	deviceBatchSizes = NULL;
	deviceLengths = NULL;
	devices = NULL;
	hostDevices = NULL;
}

void StreamEnvironment::allocateDeviceMemory() {
	if (numDevices <= 0) {
		return;
	}
	cudaError_t err = cudaMallocHost(&devices, numDevices * sizeof(float**));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	err = cudaMallocHost(&hostDevices, numDevices * sizeof(float**));
	if (err != cudaSuccess) {
		throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
	}
	for (int i = 0; i < numDevices; i++) {
		if (deviceBatchSizes[i] <= 0) {
			devices[i] = NULL;
			hostDevices[i] = NULL;
		}
		else {
			err = cudaMallocHost(&hostDevices[i], deviceBatchSizes[i] * sizeof(float*));
			if (err != cudaSuccess) {
				throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
			}
			err = cudaMalloc(&devices[i], deviceBatchSizes[i] * sizeof(float*));
			if (err != cudaSuccess) {
				throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
			}
			for (int j = 0; j < deviceBatchSizes[i]; j++) {
				err = cudaMalloc(&hostDevices[i][j], deviceLengths[i] * sizeof(float));
				if (err != cudaSuccess) {
					throw runtime_error(string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
				}
			}
			err = cudaMemcpy(devices[i], hostDevices[i], deviceBatchSizes[i] * sizeof(float*), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				throw runtime_error(string("CUDA memory copy failed: ") + cudaGetErrorString(err));
			}
		}
	}
}

long long StreamEnvironment::getDeviceMemory() {
	long long sum = 0;
	for (int i = 0; i < numDevices; i++) {
		sum += deviceBatchSizes[i] * deviceLengths[i] * sizeof(float);
	}
	return sum;
}