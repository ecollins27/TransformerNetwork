#pragma once
#include "Layer1D.h"
#include "Activation.h"

class BatchNormalization1D : public Layer1D {

public:
	const static string LAYER_NAME;
	Layer1D* prevLayer = NULL;

	float momentum;
	Matrix mean, batchMean;
	Matrix variance, batchVariance;
	Matrix std;
	Matrix parameters;
	Matrix parameterGradient;

	Optimizer<Matrix>* optimizer;

	BatchNormalization1D(float momentum);
	~BatchNormalization1D();

	void initPropagationQueue(OperationQueue& queue);
	void initBackPropQueue(OperationQueue& queue);
	void initApplicationQueue(OperationQueue& queue, float learningRate, int& t);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);

	void initPredictQueue(OperationQueue& queue);
	void setOptimizer(Optimizer<>* optimizer);
	int getNumParameters();
};

class BatchMeanOperation : public DBinary<Matrix, Matrix> {

public:
	BatchMeanOperation(Matrix& A, Matrix& B) : DBinary<Matrix, Matrix>(A, B) { return; };
	bool operate(OperationQueue* queue, int threadID);
};

class BatchVarianceOperation : public DTrinary<Matrix, Matrix, Matrix> {

public:
	BatchVarianceOperation(Matrix& A, Matrix& B, Matrix& C) : DTrinary<Matrix, Matrix, Matrix>(A, B, C) { return; };
	bool operate(OperationQueue* queue, int threadID);
};

class BatchNormalizationOperation : public Dnary<Matrix, Matrix> {

public:

	BatchNormalizationOperation(Matrix& A, Matrix& mean, Matrix& std, Matrix& B) : Dnary<Matrix, Matrix>(4, 1) {
		this->in[0] = &A;
		this->in[1] = &mean;
		this->in[2] = &mean;
		this->in[3] = &std;
		this->out[0] = &B;
	}
	bool operate(OperationQueue* queue, int threadID);
};

class BatchBackPropOperation : public Dnary<Matrix, Matrix> {

public:
	float c;
	BatchBackPropOperation(float c, Matrix& prevNeurons, Matrix& neuronGradient, Matrix& batchMean, Matrix& mean, Matrix& variance, Matrix& std, Matrix& parameters, Matrix& prevNeuronGradient, Matrix& parameterGradient) : Dnary<Matrix, Matrix>(7, 2) {
		this->c = c;
		this->in[0] = &prevNeurons;
		this->in[1] = &neuronGradient;
		this->in[2] = &batchMean;
		this->in[3] = &mean;
		this->in[4] = &variance;
		this->in[5] = &std;
		this->in[6] = &parameters;
		this->out[0] = &prevNeuronGradient;
		this->out[1] = &parameterGradient;
	}
	bool operate(OperationQueue* queue, int threadID);
};

