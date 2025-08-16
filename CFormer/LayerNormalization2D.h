#pragma once
#include "Layer2D.h"

class LayerNormalization2D : public Layer2D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer = NULL;

	Matrix* means;
	Matrix* variances;
	Matrix* stds;

	void initPropagationQueue(OperationQueue& queue);
	void initBackPropQueue(OperationQueue& queue);
	void setPrevLayer(Layer* prevLayer);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);

	void setBatchSize(int batchSize);
};

class LayerNormalizationOperation : public Dnary<Matrix, Matrix> {

public:
	LayerNormalizationOperation(Matrix& input, Matrix& mean, Matrix& variance, Matrix& std, Matrix& output) : Dnary<Matrix, Matrix>(1, 4) {
		this->in[0] = &input;
		this->out[0] = &mean;
		this->out[1] = &variance;
		this->out[2] = &std;
		this->out[3] = &output;
	};
	bool operate(OperationQueue* queue, int threadID);
};

class LayerNormalizationBackPropOperation : public Dnary<Matrix, Matrix> {

public:
	LayerNormalizationBackPropOperation(Matrix& input, Matrix& outputGrad, Matrix& mean, Matrix& variance, Matrix& std, Matrix& inputGrad) : Dnary<Matrix, Matrix>(5, 1) {
		this->in[0] = &input;
		this->in[1] = &outputGrad;
		this->in[2] = &mean;
		this->in[3] = &variance;
		this->in[4] = &std;
		this->out[0] = &inputGrad;
	};
	bool operate(OperationQueue* queue, int threadID);
};

