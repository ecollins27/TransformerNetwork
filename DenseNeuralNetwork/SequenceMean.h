#pragma once
#include "Layer1D.h"
#include "Layer2D.h"
#include <atomic>

class SequenceMean : public Layer1D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer;

	atomic<int> forwardThreadCount, backThreadCount;
	atomic<bool> gradientCalculated;

	Activation* activation;

	Matrix means;
	Matrix backPropIntermediate;

	SequenceMean(Activation* activation);
	~SequenceMean();

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);
};

