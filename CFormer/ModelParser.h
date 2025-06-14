#pragma once
#include <fstream>
#include "Activation.h"
#include "BatchNormalization1D.h"
#include "Dense1D.h"
#include "Dropout1D.h"
#include "Gated1D.h"
#include "Input1D.h"
#include "ResidualAdd1D.h"
#include "ResidualSave1D.h"
#include "SequenceMean.h"

#include "Dense2D.h"
#include "Dropout2D.h"
#include "Gated2D.h"
#include "Input2D.h"
#include "LayerNormalization2D.h"
#include "LinformerAttention.h"
#include "TransformerAttention.h"
#include "PositionalEncoding2D.h"
#include "ResidualAdd2D.h"
#include "ResidualSave2D.h"
#include "Model1D.h"
#include "Model2DTo1D.h"

class ModelParser {

public:
	static float getNextFloat(string line, int* commaIndex, int* newCommaIndex);
	static int getNextInt(string line, int* commaIndex, int* newCommaIndex);
	static string getNextString(string& line, int* commaIndex, int* newCommaIndex);
	static void getNextLine(ifstream& file, string& line, int* commaIndex, int* newCommaIndex);
	static Activation* readActivation(string& line, int* commaIndex, int* newCommaIndex);
	static void addLayer(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize, bool* layer1D);
	static Model* parseModel(string fileName);
};

