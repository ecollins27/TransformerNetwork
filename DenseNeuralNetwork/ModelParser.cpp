#include "ModelParser.h"

float ModelParser::getNextFloat(string line, int* commaIndex, int* newCommaIndex) {
	float value = stof(line.substr(*commaIndex + 1, *newCommaIndex - *commaIndex - 1));
	*commaIndex = *newCommaIndex;
	*newCommaIndex = line.find_first_of(",", *commaIndex + 1);
	return value;
}

int ModelParser::getNextInt(string line, int* commaIndex, int* newCommaIndex) {
	int value = stoi(line.substr(*commaIndex + 1, *newCommaIndex - *commaIndex - 1));
	*commaIndex = *newCommaIndex;
	*newCommaIndex = line.find_first_of(",", *commaIndex + 1);
	return value;
}

string ModelParser::getNextString(string& line, int* commaIndex, int* newCommaIndex) {
	string value = line.substr(*commaIndex + 1, *newCommaIndex - *commaIndex - 1);
	*commaIndex = *newCommaIndex;
	*newCommaIndex = line.find_first_of(",", *commaIndex + 1);
	return value;
}

void ModelParser::getNextLine(ifstream& file, string& line, int* commaIndex, int* newCommaIndex) {
	getline(file, line);
	*commaIndex = -1;
	*newCommaIndex = line.find_first_of(",", *commaIndex + 1);
}

Activation* ModelParser::readActivation(string& line, int* commaIndex, int* newCommaIndex) {
	string activationName = getNextString(line, commaIndex, newCommaIndex);
	if (activationName.compare("Sigmoid") == 0) {
		return { new Sigmoid() };
	}
	else if (activationName.compare("Relu") == 0) {
		return { new Relu() };
	}
	else if (activationName.compare("Elu") == 0) {
		return { new Elu(getNextFloat(line,commaIndex, newCommaIndex)) };
	}
	else if (activationName.compare("Selu") == 0) {
		return { new Selu() };
	}
	else if (activationName.compare("Tanh") == 0) {
		return { new Tanh() };
	}
	else if (activationName.compare("Swish") == 0) {
		return { new Swish() };
	}
	else if (activationName.compare("Softmax") == 0) {
		return { new Softmax() };
	}
	else {
		return { new None() };
	}
}

void ModelParser::addLayer(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize, bool* layer1D) {
	string layerName = getNextString(line, commaIndex, newCommaIndex);
	if (layerName.compare(BatchNormalization1D::LAYER_NAME) == 0) {
		BatchNormalization1D::load(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare(Dense1D::LAYER_NAME) == 0) {
		Dense1D::load(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare(Dropout1D::LAYER_NAME) == 0) {
		Dropout1D::load(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare(Gated1D::LAYER_NAME) == 0) {
		Gated1D::load(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare(ResidualAdd1D::LAYER_NAME) == 0) {
		ResidualAdd1D::load(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare(ResidualSave1D::LAYER_NAME) == 0) {
		ResidualSave1D::load(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare(SequenceMean::LAYER_NAME) == 0) {
		SequenceMean::load(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare(Dense2D::LAYER_NAME) == 0) {
		Dense2D::load(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare(Dropout2D::LAYER_NAME) == 0) {
		Dropout2D::load(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare(Gated2D::LAYER_NAME) == 0) {
		Gated2D::load(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare(LayerNormalization2D::LAYER_NAME) == 0) {
		LayerNormalization2D::load(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare(LinformerAttention::LAYER_NAME) == 0) {
		LinformerAttention::load(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare(TransformerAttention::LAYER_NAME) == 0) {
		TransformerAttention::load(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare(PositionalEncoding2D::LAYER_NAME) == 0) {
		PositionalEncoding2D::load(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare(ResidualAdd2D::LAYER_NAME) == 0) {
		ResidualAdd2D::load(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare(ResidualSave2D::LAYER_NAME) == 0) {
		ResidualSave2D::load(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else {
		throw invalid_argument("Cannot parse layer type");
	}
}

Model* ModelParser::parseModel(string filename) {
	string line;
	ifstream file(filename);
	getline(file, line);
	string modelType = line;
	getline(file, line);
	int inputSize = stoi(line);
	Model* model = NULL;
	bool* layer1D = new bool[1] {true};
	if (modelType.compare(Model1D::MODEL_NAME) == 0) {
		model = { new Model1D(inputSize) };
	}
	else if (modelType.compare(Model2DTo1D::MODEL_NAME) == 0) {
		model = { new Model2DTo1D(inputSize) };
		*layer1D = false;
	} else {
		throw invalid_argument("Unable to parse model type");
	}
	int prevSize = inputSize + 1;
	int commaIndex;
	int newCommaIndex;
	while (getline(file, line)) {
		commaIndex = -1;
		newCommaIndex = line.find_first_of(",", commaIndex + 1);
		addLayer(model, file, line, &commaIndex, &newCommaIndex, &prevSize, layer1D);
	}
	return model;
}