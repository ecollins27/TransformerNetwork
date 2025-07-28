#pragma once
#include "Matrix2.h"
#include "MatrixBatch.h"
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include "Operation.h"


class Comparator {
public:
	bool operator()(Operation* o1, Operation* o2) {
		return o1->getPrereqsUnmet() > o2->getPrereqsUnmet();
	}
};
class PropagationQueue {

public:


	int numThreads;
	thread* threads;
	priority_queue<Operation*, vector<Operation*>, Comparator> operationQueue;
	vector<Operation*> operations;
	mutex queueLock;
	atomic<bool>* deviceLocks;

	int numDevices;
	int* deviceBatchSizes;
	int* deviceLengths;
	float**** devices; // numThreads * numDevices * deviceBatchSizes * deviceLengths
	float**** hostDevices;

	PropagationQueue(int numStreams);
	void threadRun(int threadID);
	void run();
	void reset();
	void enqueueOperation(Operation* operation);
	void finalize();
	void allocateDeviceMemory();
	long long getDeviceMemory();
	int getNextAvailableThread();
};