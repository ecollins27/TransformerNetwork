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
#include "MultiHeadAttention.h"
#include "PositionalEncoding2D.h"
#include "ResidualAdd2D.h"
#include "ResidualSave2D.h"
#include "Model1D.h"
#include "Model2D.h"

class ModelParser {

public:
	static void* parseModel(string fileName);
};

