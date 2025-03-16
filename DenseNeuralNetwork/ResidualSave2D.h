#pragma once
#include "Layer2D.h"

class ResidualSave2D : public Layer2D {

public:
	const static string LAYER_NAME;

	Layer2D* prevLayer;

	void propagateLayer(int num);
	void backPropagate(int num);
	void setPrevLayer(Layer* prevLayer);
	void save(ofstream& file);
	static void load(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize);

	void backPropagateWithResidual(int num);
};

