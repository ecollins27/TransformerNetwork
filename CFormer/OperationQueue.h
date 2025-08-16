#pragma once
#include "Matrix.h"
#include "MatrixBatch.h"
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include "Operation.h"

class OperationQueue {

public:


	int numThreads;
	thread* threads;
	cudaStream_t* streams;
	cublasHandle_t* handles;
	vector<Operation*> operations;
	atomic<bool>* deviceLocks;
	atomic<int> devicesUsed;
	mutex lock;

	int numDevices;
	int* deviceBatchSizes;
	int* deviceLengths;
	float**** devices; // numThreads * numDevices * deviceBatchSizes * deviceLengths
	float**** hostDevices;

	OperationQueue(int numStreams);
	void debugCall() {};
	int getMinIndex(vector<Operation*> v);
	bool operationsAllocated(vector<Operation*> v);
	void threadRun(int threadID);
	void run();
	void reset();
	void enqueue(Operation* operation);
	void finalize();
	void allocateDeviceMemory();
	long long getDeviceMemory();
	int getNextAvailableThread();
};