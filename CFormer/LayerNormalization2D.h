#pragma once
#include "Layer2D.h"

class LayerNormalization2D : public Layer2D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer = NULL;

	Matrix2 mean;
	Matrix2 variance;
	Matrix2 std;

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);

	void setBatchSize(int batchSize);
};

