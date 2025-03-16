#pragma once
#include "Layer1D.h"
#include "ResidualSave1D.h"

class ResidualAdd1D : public Layer1D {

public:
	const static string LAYER_NAME;

	Layer1D* prevLayer;

	ResidualSave1D* residual;

	ResidualAdd1D(ResidualSave1D* residualLayer);
	~ResidualAdd1D();

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void setBatchSize(int batchSize);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);
};

