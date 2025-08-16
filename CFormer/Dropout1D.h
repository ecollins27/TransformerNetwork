#pragma once
#include "Layer1D.h"

class Dropout1D : public Layer1D {

public:
	const static string LAYER_NAME;

	Layer1D* prevLayer = NULL;

	float dropoutRate;
	bool** dropped;

	Dropout1D(float dropoutRate);
	~Dropout1D();

	void initPropagationQueue(OperationQueue& queue);
	void initBackPropQueue(OperationQueue& queue);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);

	virtual void initPredictQueue(OperationQueue& queue);
};

template<typename Type>
class DropoutForwardPropOperation : public HBinary<Type, Type> {

public:
	float dropoutRate;
	bool** dropped;
	uniform_real_distribution<float> distribution;
	default_random_engine generator;

	DropoutForwardPropOperation(float dropoutRate, Type& input, Type& output, bool** dropped) : HBinary<Type, Type>(input, output) {
		this->dropped = dropped;
		this->dropoutRate = dropoutRate;
		this->distribution = uniform_real_distribution<float>(0, 1);
	};
	bool operate(OperationQueue* queue, int threadID);
};

template<typename Type>
class DropoutBackPropOperation : public HBinary<Type, Type> {

public:
	float dropoutRate;
	bool** dropped;

	DropoutBackPropOperation(float dropoutRate, Type& input, Type& output, bool** dropped) : HBinary<Type, Type>(input, output) {
		this->dropped = dropped;
		this->dropoutRate = dropoutRate;
	};
	bool operate(OperationQueue* queue, int threadID);
};

