#pragma once
#include "Layer1D.h"
#include "Layer2D.h"
#include <atomic>

class SequenceMean : public Layer1D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer = NULL;

	atomic<int> forwardThreadCount, backThreadCount;
	atomic<bool> gradientCalculated;

	Activation* activation;

	Matrix means;
	Matrix backPropIntermediate;

	SequenceMean(Activation* activation);
	~SequenceMean();

	void initPropagationQueue(OperationQueue& queue);
	void initBackPropQueue(OperationQueue& queue);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);
};

class MeanCondenseOperation : public Dnary<Matrix, Matrix> {

public:
	MeanCondenseOperation(int batchSize, Matrix*& A, Matrix& B) : Dnary<Matrix, Matrix>(batchSize, 1) { return; };
	bool operate(OperationQueue* queue, int threadID);
};

class MeanCondenseBackPropOperation : public Dnary<Matrix, Matrix> {

public:
	MeanCondenseBackPropOperation(int batchSize, Matrix& A, Matrix*& B) : Dnary<Matrix, Matrix>(1, batchSize) { return; };
	bool operate(OperationQueue* queue, int threadID);
};

