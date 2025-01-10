#include "ModelParser.h"

float getNextFloat(string line, int* commaIndex, int* newCommaIndex) {
	float value = stof(line.substr(*commaIndex + 1, *newCommaIndex - *commaIndex - 1));
	*commaIndex = *newCommaIndex;
	*newCommaIndex = line.find_first_of(",", *commaIndex + 1);
	return value;
}

int getNextInt(string line, int* commaIndex, int* newCommaIndex) {
	int value = stoi(line.substr(*commaIndex + 1, *newCommaIndex - *commaIndex - 1));
	*commaIndex = *newCommaIndex;
	*newCommaIndex = line.find_first_of(",", *commaIndex + 1);
	return value;
}

string getNextString(string& line, int* commaIndex, int* newCommaIndex) {
	string value = line.substr(*commaIndex + 1, *newCommaIndex - *commaIndex - 1);
	*commaIndex = *newCommaIndex;
	*newCommaIndex = line.find_first_of(",", *commaIndex + 1);
	return value;
}

void getNextLine(ifstream& file, string& line, int* commaIndex, int* newCommaIndex) {
	getline(file, line);
	*commaIndex = -1;
	*newCommaIndex = line.find_first_of(",", *commaIndex + 1);
}

Activation* readActivation(string& line, int* commaIndex, int* newCommaIndex) {
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

void addDenseLayer(void* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize, bool* layer1D) {
	Activation* activation = readActivation(line, commaIndex, newCommaIndex);
	int size = getNextInt(line, commaIndex, newCommaIndex);
	if (*layer1D) {
		Dense1D* denseLayer = new Dense1D(activation, size);
		((Model1D*)nn)->addLayer(denseLayer);
		for (int i = 0; i < size; i++) {
			getNextLine(file, line, commaIndex, newCommaIndex);
			for (int j = 0; j < *prevSize; j++) {
				denseLayer->weights.r(i, j) = getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
	}
	else {
		Dense2D* denseLayer = new Dense2D(activation, size);
		((Model2D*)nn)->addLayer(denseLayer);
		for (int i = 0; i < size; i++) {
			getNextLine(file, line, commaIndex, newCommaIndex);
			for (int j = 0; j < *prevSize; j++) {
				denseLayer->weights.r(i, j) = getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
	}
	*prevSize = size + 1;
}

void addBatchNormalization(void* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	BatchNormalization1D* batchNormalization = { new BatchNormalization1D(0.9) };
	((Model1D*)nn)->addLayer(batchNormalization);
	for (int i = 0; i < 2; i++) {
		getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize - 1; j++) {
			batchNormalization->parameters.r(i, j) = getNextFloat(line, commaIndex, newCommaIndex);
		}
	}
	getNextLine(file, line, commaIndex, newCommaIndex);
	for (int j = 0; j < *prevSize - 1; j++) {
		batchNormalization->mean.r(0, j) = getNextFloat(line, commaIndex, newCommaIndex);
	}
	getNextLine(file, line, commaIndex, newCommaIndex);
	for (int j = 0; j < *prevSize - 1; j++) {
		batchNormalization->variance.r(0, j) = getNextFloat(line, commaIndex, newCommaIndex);
		batchNormalization->std.r(0, j) = sqrt(batchNormalization->variance(0, j) + 0.0000001);
	}
}

void addDropout(void* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize, bool* layer1D) {
	if (*layer1D) {
		Dropout1D* dropout = { new Dropout1D(getNextFloat(line, commaIndex, newCommaIndex)) };
		((Model1D*)nn)->addLayer(dropout);
	}
	else {
		Dropout2D* dropout = { new Dropout2D(getNextFloat(line, commaIndex, newCommaIndex)) };
		((Model2D*)nn)->addLayer(dropout);
	}
}

void addGatedLayer(void* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize, bool* layer1D) {
	Activation* activation = readActivation(line, commaIndex, newCommaIndex);
	int size = getNextInt(line, commaIndex, newCommaIndex);
	if (*layer1D) {
		Gated1D* gatedLayer = { new Gated1D(activation, size) };
		((Model1D*)nn)->addLayer(gatedLayer);
		for (int i = 0; i < size; i++) {
			getNextLine(file, line, commaIndex, newCommaIndex);
			for (int j = 0; j < *prevSize; j++) {
				gatedLayer->weights1.r(i, j) = getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
		for (int i = 0; i < size; i++) {
			getNextLine(file, line, commaIndex, newCommaIndex);
			for (int j = 0; j < *prevSize; j++) {
				gatedLayer->weights2.r(i, j) = getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
	}
	else {
		Gated2D* gatedLayer = { new Gated2D(activation, size) };
		((Model2D*)nn)->addLayer(gatedLayer);
		for (int i = 0; i < size; i++) {
			getNextLine(file, line, commaIndex, newCommaIndex);
			for (int j = 0; j < *prevSize; j++) {
				gatedLayer->weights1.r(i, j) = getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
		for (int i = 0; i < size; i++) {
			getNextLine(file, line, commaIndex, newCommaIndex);
			for (int j = 0; j < *prevSize; j++) {
				gatedLayer->weights2.r(i, j) = getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
	}
	*prevSize = size + 1;
}

void addLayerNormalization(void* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	LayerNormalization2D* layerNormalization = { new LayerNormalization2D() };
	((Model2D*)nn)->addLayer(layerNormalization);
}

void addMultiHeadAttentionLayer(void* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	int numHeads = getNextInt(line, commaIndex, newCommaIndex);
	int keySize = getNextInt(line, commaIndex, newCommaIndex);
	int valueSize = getNextInt(line, commaIndex, newCommaIndex);
	MultiHeadAttention* multiHeadAttentionLayer = { new MultiHeadAttention(numHeads, keySize, valueSize) };
	((Model2D*)nn)->addLayer(multiHeadAttentionLayer);
	for (int i = 0; i < numHeads; i++) {
		for (int j = 0; j < keySize; j++) {
			getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < *prevSize; k++) {
				multiHeadAttentionLayer->Wq[i].r(j, k) = getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
		for (int j = 0; j < keySize; j++) {
			getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < *prevSize; k++) {
				multiHeadAttentionLayer->Wk[i].r(j, k) = getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
		for (int j = 0; j < valueSize; j++) {
			getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < *prevSize; k++) {
				multiHeadAttentionLayer->Wv[i].r(j, k) = getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
	}
	for (int i = 0; i < *prevSize - 1; i++) {
		getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < numHeads * valueSize; j++) {
			multiHeadAttentionLayer->Wo.r(i, j) = getNextFloat(line, commaIndex, newCommaIndex);
		}
	}
}

void addResidualAdd(void* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize, bool* layer1D) {
	int saveIndex = getNextInt(line, commaIndex, newCommaIndex);
	if (*layer1D) {
		ResidualAdd1D* residualAdd = { new ResidualAdd1D(((Model1D*)nn)->getLayer<ResidualSave1D>(saveIndex))};
		((Model1D*)nn)->addLayer(residualAdd);
	}
	else {
		ResidualAdd2D* residualAdd = { new ResidualAdd2D(((Model2D*)nn)->getLayer<ResidualSave2D>(saveIndex)) };
		((Model2D*)nn)->addLayer(residualAdd);
	}
}

void addResidualSave(void* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize, bool* layer1D) {
	if (*layer1D) {
		ResidualSave1D* residualSave = { new ResidualSave1D() };
		((Model1D*)nn)->addLayer(residualSave);
	}
	else {
		ResidualSave2D* residualSave = { new ResidualSave2D() };
		((Model2D*)nn)->addLayer(residualSave);
	}
}

void addPositionalEncodingLayer(void* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	int L = getNextInt(line, commaIndex, newCommaIndex);
	PositionalEncoding2D* positionalEncodingLayer = { new PositionalEncoding2D(L) };
	((Model2D*)nn)->addLayer(positionalEncodingLayer);
}

void addSequenceMean(void* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Activation* activation = readActivation(line, commaIndex, newCommaIndex);
	SequenceMean* batchSum = { new SequenceMean(activation) };
	((Model2D*)nn)->addLayer(batchSum);
}

void addSavedLayer(void* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize, bool* layer1D) {
	string layerName = getNextString(line, commaIndex, newCommaIndex);
	if (layerName.compare("DenseLayer") == 0) {
		addDenseLayer(nn, file, line, commaIndex, newCommaIndex, prevSize, layer1D);
	}
	else if (layerName.compare("BatchNormalization") == 0) {
		addBatchNormalization(nn, file, line, commaIndex, newCommaIndex, prevSize);
	}
	else if (layerName.compare("Dropout") == 0) {
		addDropout(nn, file, line, commaIndex, newCommaIndex, prevSize, layer1D);
	}
	else if (layerName.compare("GatedLayer") == 0) {
		addGatedLayer(nn, file, line, commaIndex, newCommaIndex, prevSize, layer1D);
	}
	else if (layerName.compare("LayerNormalization") == 0) {
		addLayerNormalization(nn, file, line, commaIndex, newCommaIndex, prevSize);
	}
	else if (layerName.compare("MultiHeadAttention") == 0) {
		addMultiHeadAttentionLayer(nn, file, line, commaIndex, newCommaIndex, prevSize);
	}
	else if (layerName.compare("ResidualAdd") == 0) {
		addResidualAdd(nn, file, line, commaIndex, newCommaIndex, prevSize, layer1D);
	}
	else if (layerName.compare("ResidualSave") == 0) {
		addResidualSave(nn, file, line, commaIndex, newCommaIndex, prevSize, layer1D);
	}
	else if (layerName.compare("PositionalEncodingLayer") == 0) {
		addPositionalEncodingLayer(nn, file, line, commaIndex, newCommaIndex, prevSize);
	}
	else if (layerName.compare("SequenceMean") == 0) {
		addSequenceMean(nn, file, line, commaIndex, newCommaIndex, prevSize);
	}
	else {
		printf("\nFailed to Parse");
	}
}

void* ModelParser::parseModel(string filename) {
	string line;
	ifstream file(filename);
	getline(file, line);
	string modelType = line;
	getline(file, line);
	int inputSize = stoi(line);
	void* model = NULL;
	bool* layer1D = new bool[1] {true};
	if (modelType.compare("Model1D") == 0) {
		model = { new Model1D(inputSize) };
	}
	else {
		model = { new Model2D(inputSize) };
		*layer1D = false;
	}
	int prevSize = inputSize + 1;
	int commaIndex;
	int newCommaIndex;
	while (getline(file, line)) {
		commaIndex = -1;
		newCommaIndex = line.find_first_of(",", commaIndex + 1);
		addSavedLayer(model, file, line, &commaIndex, &newCommaIndex, &prevSize, layer1D);
	}
	return model;
}