#pragma once
#include "Layer2D.h"

class PositionalEncoding2D : public Layer2D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer = NULL;
	float L;

	PositionalEncoding2D(float L = 10000);

	void initPropagationQueue(OperationQueue& queue);
	void initBackPropQueue(OperationQueue& queue);
	void setPrevLayer(Layer* prevLayer);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);
};

class PositionalEncodingOperation : public DBinary<Matrix, Matrix> {

public:
	float L;
	PositionalEncodingOperation(float L, Matrix& A, Matrix& B) : DBinary<Matrix, Matrix>(A, B) { this->L = L; };
	bool operate(OperationQueue* queue, int threadID);
};

