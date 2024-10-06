#include "NeuralNetwork.h"
#include "BatchNormalization.h"
#include "GatedLayer.h"

NeuralNetwork::NeuralNetwork(int inputSize) {
	inputLayer = { new InputLayer(inputSize) };
	outputLayer = inputLayer;
	t = 0;
}

double getNextDouble(string line, int* commaIndex, int* newCommaIndex) {
	double value = stod(line.substr(*commaIndex + 1, *newCommaIndex - *commaIndex - 1));
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
	} else if (activationName.compare("Relu") == 0) {
		return { new Relu() };
	} else if (activationName.compare("Elu") == 0) {
		return { new Elu(getNextDouble(line,commaIndex, newCommaIndex)) };
	} else if (activationName.compare("Selu") == 0) {
		return { new Selu() };
	} else if (activationName.compare("Tanh") == 0) {
		return { new Tanh() };
	} else if (activationName.compare("Swish") == 0) {
		return { new Swish() };
	} else if (activationName.compare("Softmax") == 0) {
		return { new Softmax() };
	} else {
		return { new None() };
	}
}

void addDenseLayer(NeuralNetwork* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Activation* activation = readActivation(line, commaIndex, newCommaIndex);
	int size = getNextInt(line, commaIndex, newCommaIndex);
	DenseLayer* denseLayer = { new DenseLayer(activation, size) };
	nn->addLayer(denseLayer);
	for (int i = 0; i < size; i++) {
		getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize; j++) {
			denseLayer->weights[i][j] = getNextDouble(line, commaIndex, newCommaIndex);
		}
	}
	*prevSize = size + 1;
}

void addBatchNormalization(NeuralNetwork* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	BatchNormalization* batchNormalization = { new BatchNormalization() };
	nn->addLayer(batchNormalization);
	for (int i = 0; i < 2; i++) {
		getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize - 1; j++) {
			batchNormalization->parameters[i][j] = getNextDouble(line, commaIndex, newCommaIndex);
		}
	}
	getNextLine(file, line, commaIndex, newCommaIndex);
	for (int j = 0; j < *prevSize - 1; j++) {
		batchNormalization->mean[0][j] = getNextDouble(line, commaIndex, newCommaIndex);
	}
	getNextLine(file, line, commaIndex, newCommaIndex);
	for (int j = 0; j < *prevSize - 1; j++) {
		batchNormalization->variance[0][j] = getNextDouble(line, commaIndex, newCommaIndex);
		batchNormalization->std[0][j] = sqrt(batchNormalization->variance[0][j] + 0.0000001);
	}
}

void addDropout(NeuralNetwork* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Dropout* dropout = { new Dropout(getNextDouble(line, commaIndex, newCommaIndex)) };
	nn->addLayer(dropout);
}

void addGatedLayer(NeuralNetwork* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	Activation* activation = readActivation(line, commaIndex, newCommaIndex);
	int size = getNextInt(line, commaIndex, newCommaIndex);
	GatedLayer* gatedLayer = { new GatedLayer(activation, size) };
	nn->addLayer(gatedLayer);
	for (int i = 0; i < size; i++) {
		getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize; j++) {
			gatedLayer->weights1[i][j] = getNextDouble(line, commaIndex, newCommaIndex);
		}
	}
	for (int i = 0; i < size; i++) {
		getNextLine(file, line, commaIndex, newCommaIndex);
		for (int j = 0; j < *prevSize; j++) {
			gatedLayer->weights2[i][j] = getNextDouble(line, commaIndex, newCommaIndex);
		}
	}
	*prevSize = size + 1;
}

void addSavedLayer(NeuralNetwork* nn, ifstream& file, string& line, int* commaIndex, int* newCommaIndex, int* prevSize) {
	string layerName = getNextString(line, commaIndex, newCommaIndex);
	if (layerName.compare("DenseLayer") == 0) {
		addDenseLayer(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare("BatchNormalization") == 0) {
		addBatchNormalization(nn, file, line, commaIndex, newCommaIndex, prevSize);
	} else if (layerName.compare("Dropout") == 0) {
		addDropout(nn, file, line, commaIndex, newCommaIndex, prevSize);
	}
}


NeuralNetwork::NeuralNetwork(string fileName) {
	string line;
	ifstream file(fileName);
	getline(file, line);
	int inputSize = stoi(line);
	inputLayer = { new InputLayer(inputSize) };
	outputLayer = inputLayer;
	int prevSize = inputSize + 1;
	int commaIndex;
	int newCommaIndex;
	while (getline(file, line)) {
		commaIndex = -1;
		newCommaIndex = line.find_first_of(",", commaIndex + 1);
		addSavedLayer(this, file, line, &commaIndex, &newCommaIndex, &prevSize);
	}
	t = 0;
}

void NeuralNetwork::addLayer(Layer* layer) {
	outputLayer->setNextLayer(layer);
	layer->setPrevLayer(outputLayer);
	outputLayer = layer;
}

void NeuralNetwork::predict(double** input) {
	inputLayer->setInput(input);
	inputLayer->predict();
}

void NeuralNetwork::forwardPropagate(double** input) {
	inputLayer->setInput(input);
	inputLayer->forwardPropagate();
}

void NeuralNetwork::backPropagate(Loss* lossFunction, double** yTrue) {
	lossFunction->differentiate(outputLayer, yTrue);
	outputLayer->backPropagate();
}

void NeuralNetwork::applyGradients(double learningRate) {
	t++;
	inputLayer->applyGradients(learningRate, t);
}

void NeuralNetwork::fit(Loss* lossFunction, double** X, double** y, double* losses, int numMetrics, Loss** metrics, TrainingParams* params) {
	forwardPropagate(X);
	backPropagate(lossFunction, y);
	for (int i = 0; i < numMetrics; i++) {
		losses[i] += metrics[i]->loss(outputLayer, y);
	}
	losses[numMetrics] += lossFunction->loss(outputLayer, y);
}

void NeuralNetwork::shuffle(int numData, double** X, double** y) {
	for (int i = 0; i < numData; i++) {
		int index = (int)(numData * ((double)rand() / (RAND_MAX + 1)));
		swap(X[i], X[index]);
		swap(y[i], y[index]);
	}
}

void NeuralNetwork::fit(Loss* lossFunction, int numData, double** X, double** y, int numMetrics, Loss** metrics, TrainingParams* params) {
	double valSplit = params->get<double>(TrainingParams::VAL_SPLIT);
	int batchSize = params->get<int>(TrainingParams::BATCH_SIZE);
	int numEpochs = params->get<int>(TrainingParams::NUM_EPOCHS);
	double learningRate = params->get<double>(TrainingParams::LEARNING_RATE);
	inputLayer->setOptimizer(params->get<Optimizer*>(TrainingParams::OPTIMIZER));
	double* averages = NULL;
	averages = new double[numMetrics + 1];
	int trainingNum = (int)(numData * (1 - valSplit));
	trainingNum -= trainingNum % batchSize;
	for (int epoch = 0; epoch < numEpochs; epoch++) {
		inputLayer->setBatchSize(batchSize);
		shuffle(numData, X, y);
		for (int i = 0; i < numMetrics + 1; i++) {
			averages[i] = 0;
		}
		for (int i = 0; i <= trainingNum - batchSize; i += batchSize) {
			printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, numEpochs, i, trainingNum, averages[numMetrics] / i);
			for (int j = 0; j < numMetrics; j++) {
				printf("%s:%f  ", metrics[j]->toString().c_str(), averages[j] / i);
			}
			fit(lossFunction, &X[i], &y[i], averages, numMetrics, metrics, params);
			applyGradients(learningRate);
		}
		printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, numEpochs, trainingNum, trainingNum, averages[numMetrics] / trainingNum);
		for (int j = 0; j < numMetrics; j++) {
			printf("%s:%f  ", metrics[j]->toString().c_str(), averages[j] / trainingNum);
		}
		int valNum = (int)(numData * valSplit);
		if (valNum > 0) {
			for (int i = 0; i < numMetrics + 1; i++) {
				averages[i] = 0;
			}
			int i = trainingNum;
			for (i = trainingNum; i <= numData - batchSize; i += batchSize) {
				predict(&X[i]);
				averages[numMetrics] += lossFunction->loss(outputLayer, &y[i]);
				for (int j = 0; j < numMetrics; j++) {
					averages[j] += metrics[j]->loss(outputLayer, &y[i]);
				}
			}
			int remaining = numData - i;
			if (remaining > 0) {
				inputLayer->setBatchSize(remaining);
				predict(&X[i]);
				averages[numMetrics] += lossFunction->loss(outputLayer, &y[i]);
				for (int j = 0; j < numMetrics; j++) {
					averages[j] += metrics[j]->loss(outputLayer, &y[i]);
				}
			}
			printf("ValLoss:%f  ", averages[numMetrics] / (numData - trainingNum));
			for (i = 0; i < numMetrics; i++) {
				printf("Val%s:%f  ", metrics[i]->toString().c_str(), averages[i] / (numData - trainingNum));
			}
		}
		printf("\n");
	}
}

void NeuralNetwork::test(Loss* lossFunction, int numData, double** X, double** y, int numMetrics, Loss** metrics) {
	inputLayer->setBatchSize(16);
	double* averages = new double[numMetrics + 1];
	for (int i = 0; i <= numData - 16; i += 16) {
		predict(&X[i]);
		averages[numMetrics] += lossFunction->loss(outputLayer, &y[i]);
		for (int j = 0; j < numMetrics; j++) {
			averages[j] += metrics[j]->loss(outputLayer, &y[i]);
		}
	}
	if (numData % 16 > 0) {
		inputLayer->setBatchSize(numData % 16);
		predict(&X[numData - numData % 16]);
		averages[numMetrics] += lossFunction->loss(outputLayer, &y[numData - numData % 16]);
		for (int j = 0; j < numMetrics; j++) {
			averages[j] += metrics[j]->loss(outputLayer, &y[numData - numData % 16]);
		}
	}
	printf("TestLoss:%f  ", averages[numMetrics] / numData);
	for (int j = 0; j < numMetrics; j++) {
		printf("Test%s:%f  ", metrics[j]->toString().c_str(), averages[j] / numData);
	}
	printf("\n");
}

void NeuralNetwork::setTrainable(bool trainable) {
	inputLayer->setTrainable(trainable);
}

void NeuralNetwork::save(string fileName) {
	ofstream file(fileName.c_str());
	inputLayer->save(file);
	file.close();
}

int NeuralNetwork::getNumParameters() {
	return inputLayer->getNumParameters();
}