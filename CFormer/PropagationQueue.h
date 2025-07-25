#pragma once
#include "Matrix2.h"
#include "MatrixBatch.h"
#include <functional>

class StreamEnvironment;
class Operation;

class PropagationQueue {

public:


	int numStreams;
	StreamEnvironment* streams;
	vector<Operation*> operations;

	PropagationQueue(int numStreams);
	void start();
	void enqueueOperation(Operation* operation);
	long long getDeviceMemory();
};

class StreamEnvironment {
public:
	int numDevices;
	int* deviceBatchSizes;
	int* deviceLengths;
	float*** devices; //numDevices x deviceBatchSizes x deviceLengths
	float*** hostDevices;
	cudaStream_t stream;
	cublasHandle_t handle;

	StreamEnvironment();
	void allocateDeviceMemory();
	long long getDeviceMemory();
};

class Operation {

public:
	void* output;
	virtual void operate(StreamEnvironment stream) = 0;
	virtual void applyToStream(StreamEnvironment stream) = 0;
	virtual void copyToDevice(StreamEnvironment stream, int completedIndex) = 0;
	virtual void copyToHost(StreamEnvironment stream) = 0;
	virtual void findPrereqs(vector<Operation*> operations) = 0;
};