#pragma once
#include "Layer1D.h"
#include "Activation.h"

class BatchNormalization1D : public Layer1D {

public:
	const static string LAYER_NAME;
	Layer1D* prevLayer = NULL;

	float momentum;
	Matrix2 mean, batchMean;
	Matrix2 variance, batchVariance;
	Matrix2 std;
	Matrix2 parameters;
	Matrix2 parameterGradient;

	Optimizer* optimizer;

	BatchNormalization1D(float momentum);
	~BatchNormalization1D();

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
	void setMaxMatrixLengths();
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);

	void predict(int num);
	void applyGradients(float learningRate, int t);
	void setOptimizer(Optimizer* optimizer);
	int getNumParameters();
};

