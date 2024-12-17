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

void addDenseLayer(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Activation* activation = readActivation(line, commaIndex, newCommaIndex);
	int size = getNextInt(line, commaIndex, newCommaIndex);
	DenseLayer* denseLayer = { new DenseLayer(activation, size) };
	nn->addLayer(denseLayer);
	for (int i = 0; i < size; i++) {
		getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize; j++) {
			denseLayer->weights[i][j] = getNextFloat(line, commaIndex, newCommaIndex);
		}
	}
	*prevSize = size + 1;
}

void addBatchNormalization(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	BatchNormalization* batchNormalization = { new BatchNormalization() };
	nn->addLayer(batchNormalization);
	for (int i = 0; i < 2; i++) {
		getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize - 1; j++) {
			batchNormalization->parameters[i][j] = getNextFloat(line, commaIndex, newCommaIndex);
		}
	}
	getNextLine(file, line, commaIndex, newCommaIndex);
	for (int j = 0; j < *prevSize - 1; j++) {
		batchNormalization->mean[0][j] = getNextFloat(line, commaIndex, newCommaIndex);
	}
	getNextLine(file, line, commaIndex, newCommaIndex);
	for (int j = 0; j < *prevSize - 1; j++) {
		batchNormalization->variance[0][j] = getNextFloat(line, commaIndex, newCommaIndex);
		batchNormalization->std[0][j] = sqrt(batchNormalization->variance[0][j] + 0.0000001);
	}
}

void addDropout(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Dropout* dropout = { new Dropout(getNextFloat(line, commaIndex, newCommaIndex)) };
	nn->addLayer(dropout);
}

void addGatedLayer(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Activation* activation = readActivation(line, commaIndex, newCommaIndex);
	int size = getNextInt(line, commaIndex, newCommaIndex);
	GatedLayer* gatedLayer = { new GatedLayer(activation, size) };
	nn->addLayer(gatedLayer);
	for (int i = 0; i < size; i++) {
		getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize; j++) {
			gatedLayer->weights1[i][j] = getNextFloat(line, commaIndex, newCommaIndex);
		}
	}
	for (int i = 0; i < size; i++) {
		getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize; j++) {
			gatedLayer->weights2[i][j] = getNextFloat(line, commaIndex, newCommaIndex);
		}
	}
	*prevSize = size + 1;
}

void addLayerNormalization(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	LayerNormalization* layerNormalization = { new LayerNormalization() };
	nn->addLayer(layerNormalization);
}

void addMultiHeadAttentionLayer(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	int numHeads = getNextInt(line, commaIndex, newCommaIndex);
	int keySize = getNextInt(line, commaIndex, newCommaIndex);
	int valueSize = getNextInt(line, commaIndex, newCommaIndex);
	MultiHeadAttentionLayer* multiHeadAttentionLayer = { new MultiHeadAttentionLayer(numHeads, keySize, valueSize) };
	nn->addLayer(multiHeadAttentionLayer);
	for (int i = 0; i < numHeads; i++) {
		for (int j = 0; j < keySize; j++) {
			getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < *prevSize; k++) {
				multiHeadAttentionLayer->Wq[i][j][k] = getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
		for (int j = 0; j < keySize; j++) {
			getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < *prevSize; k++) {
				multiHeadAttentionLayer->Wk[i][j][k] = getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
		for (int j = 0; j < valueSize; j++) {
			getNextLine(file, line, commaIndex, newCommaIndex);
			for (int k = 0; k < *prevSize; k++) {
				multiHeadAttentionLayer->Wv[i][j][k] = getNextFloat(line, commaIndex, newCommaIndex);
			}
		}
	}
	for (int i = 0; i < *prevSize - 1; i++) {
		getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < numHeads * valueSize; j++) {
			multiHeadAttentionLayer->Wo[i][j] = getNextFloat(line, commaIndex, newCommaIndex);
		}
	}
}

void addResidualAdd(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	int saveIndex = getNextInt(line, commaIndex, newCommaIndex);
	ResidualAdd* residualAdd = { new ResidualAdd(nn->getLayer<ResidualSave>(saveIndex)) };
	nn->addLayer(residualAdd);
}

void addResidualSave(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	ResidualSave* residualSave = { new ResidualSave() };
	nn->addLayer(residualSave);
}

void addPositionalEncodingLayer(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	int L = getNextInt(line, commaIndex, newCommaIndex);
	PositionalEncodingLayer* positionalEncodingLayer = { new PositionalEncodingLayer(L) };
	nn->addLayer(positionalEncodingLayer);
}

void addBatchMean(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Activation* activation = readActivation(line, commaIndex, newCommaIndex);
	BatchMean* batchSum = { new BatchMean(activation) };
	nn->addLayer(batchSum);
}

void addSavedLayer(Model* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	string layerName = getNextString(line, commaIndex, newCommaIndex);
	printf("\n%s\n", layerName.c_str());
	if (layerName.compare("DenseLayer") == 0) {
		addDenseLayer(nn, file, line, commaIndex, newCommaIndex, prevSize);
	}
	else if (layerName.compare("BatchNormalization") == 0) {
		addBatchNormalization(nn, file, line, commaIndex, newCommaIndex, prevSize);
	}
	else if (layerName.compare("Dropout") == 0) {
		addDropout(nn, file, line, commaIndex, newCommaIndex, prevSize);
	}
	else if (layerName.compare("GatedLayer") == 0) {
		addGatedLayer(nn, file, line, commaIndex, newCommaIndex, prevSize);
	}
	else if (layerName.compare("LayerNormalization") == 0) {
		addLayerNormalization(nn, file, line, commaIndex, newCommaIndex, prevSize);
	}
	else if (layerName.compare("MultiHeadAttention") == 0) {
		addMultiHeadAttentionLayer(nn, file, line, commaIndex, newCommaIndex, prevSize);
	}
	else if (layerName.compare("ResidualAdd") == 0) {
		addResidualAdd(nn, file, line, commaIndex, newCommaIndex, prevSize);
	}
	else if (layerName.compare("ResidualSave") == 0) {
		addResidualSave(nn, file, line, commaIndex, newCommaIndex, prevSize);
	}
	else if (layerName.compare("PositionalEncodingLayer") == 0) {
		addPositionalEncodingLayer(nn, file, line, commaIndex, newCommaIndex, prevSize);
	}
	else if (layerName.compare("BatchMean") == 0) {
		addBatchMean(nn, file, line, commaIndex, newCommaIndex, prevSize);
	}
	else {
		printf("\nFailed to Parse");
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
	if (modelType.compare("NeuralNetwork") == 0) {
		model = { new NeuralNetwork(inputSize) };
	}
	else {
		model = { new TransformerModel(inputSize) };
	}
	int prevSize = inputSize + 1;
	int commaIndex;
	int newCommaIndex;
	while (getline(file, line)) {
		commaIndex = -1;
		newCommaIndex = line.find_first_of(",", commaIndex + 1);
		addSavedLayer(model, file, line, &commaIndex, &newCommaIndex, &prevSize);
	}
	return model;
}