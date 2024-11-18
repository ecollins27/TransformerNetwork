#pragma once
#include "Layer.h"
#include <fstream>
#include "Activation.h"
#include "Model.h"
#include "BatchNormalization.h"
#include "GatedLayer.h"
#include "NeuralNetwork.h"
#include "TransformerModel.h"
#include "LayerNormalization.h"
#include "MultiHeadAttentionLayer.h"
#include "ResidualAdd.h"
#include "ResidualSave.h"

class ModelParser {

public:
	static Model* parseModel(string fileName);
};

