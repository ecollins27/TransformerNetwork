#include "NeuralNetwork.h"
#include "BatchNormalization.h"

NeuralNetwork::NeuralNetwork(Loss* lossFunction, int inputSize) {
	this->lossFunction = lossFunction;
	inputLayer = { new InputLayer(inputSize) };
	outputLayer = inputLayer;
	t = 0;
}


NeuralNetwork::NeuralNetwork(string fileName) {
	string line;
	ifstream file(fileName);
	getline(file, line);
	int commaIndex = line.find_first_of(",");
	int index = stoi(line.substr(0,commaIndex));
	int inputSize = stoi(line.substr(commaIndex + 1, line.length()));
	this->lossFunction = Loss::ALL_LOSSES[index];
	inputLayer = { new InputLayer(inputSize) };
	outputLayer = inputLayer;
	int prevSize = inputSize + 1;
	t = 0;
	while (getline(file, line)) {
		commaIndex = line.find_first_of(",");
		string layerType = line.substr(0, commaIndex);
		if (layerType.compare("DenseLayer") == 0) {
			int newCommaIndex = line.find_first_of(",", commaIndex + 1);
			index = stoi(line.substr(commaIndex + 1, newCommaIndex));
			commaIndex = newCommaIndex;
			int size = stoi(line.substr(commaIndex + 1, line.length()));
			addLayer({ new DenseLayer(Activation::ALL_ACTIVATIONS[index], size) });
			for (int i = 0; i < size; i++) {
				commaIndex = 0;
				getline(file, line);
				for (int j = 0; j < prevSize; j++) {
					newCommaIndex = line.find_first_of(",", commaIndex + 1);
					((DenseLayer*)outputLayer)->weights[i][j] = stod(line.substr(commaIndex + 1, newCommaIndex - commaIndex - 1));
					commaIndex = newCommaIndex;
				}
			}
			prevSize = size + 1;
		} else if (layerType.compare("Dropout") == 0) {
			double dropRate = stod(line.substr(commaIndex + 1, line.length()));
			addLayer({ new Dropout(dropRate) });
		} else if (layerType.compare("BatchNormalization") == 0) {
			addLayer({ new BatchNormalization() });
			int newCommaIndex = 0;
			for (int i = 0; i < 4; i++) {
				getline(file, line);
				commaIndex = -1;
				for (int j = 0; j < outputLayer->size; j++) {
					newCommaIndex = line.find_first_of(",", commaIndex + 1);
					double param = stod(line.substr(commaIndex + 1, newCommaIndex));
					if (i < 2) {
						((BatchNormalization*)outputLayer)->parameters[i][j] = param;
						commaIndex = newCommaIndex;
					} else if (i == 3) {
						((BatchNormalization*)outputLayer)->mean[0][j] = param;
					} else {
						((BatchNormalization*)outputLayer)->variance[0][j] = param;
						((BatchNormalization*)outputLayer)->std[0][j] = sqrt(param);
					}
					commaIndex = newCommaIndex;
				}
			}
		}
	}
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

double NeuralNetwork::getLoss(double** output) {
	return lossFunction->loss(outputLayer, output);
}

void NeuralNetwork::backPropagate(double** yTrue) {
	lossFunction->differentiate(outputLayer, yTrue);
	outputLayer->backPropagate();
}

void NeuralNetwork::applyGradients(TrainingParams* params) {
	t++;
	inputLayer->applyGradients(params, t);
}

void NeuralNetwork::fit(double** X, double** y, double* losses, TrainingParams* params) {
	forwardPropagate(X);
	backPropagate(y);
	for (int i = 0; i < params->numMetrics; i++) {
		losses[i] += params->metrics[i]->loss(outputLayer, y);
	}
	losses[params->numMetrics] += lossFunction->loss(outputLayer, y);
}

void NeuralNetwork::shuffle(int numData, double** X, double** y) {
	for (int i = 0; i < numData; i++) {
		int index = (int)(numData * ((double)rand() / (RAND_MAX + 1)));
		swap(X[i], X[index]);
		swap(y[i], y[index]);
	}
}

void NeuralNetwork::fit(int numData, double** X, double** y, TrainingParams* params) {
	inputLayer->setOptimizer(params->optimizer);
	double* averages = NULL;
	averages = new double[params->numMetrics + 1];
	int trainingNum = (int)(numData * (1 - params->valSplit));
	trainingNum -= trainingNum % params->batchSize;
	for (int epoch = 0; epoch < params->numEpochs; epoch++) {
		inputLayer->setBatchSize(params->batchSize);
		shuffle(numData, X, y);
		for (int i = 0; i < params->numMetrics + 1; i++) {
			averages[i] = 0;
		}
		for (int i = 0; i < trainingNum - params->batchSize; i += params->batchSize) {
			printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, params->numEpochs, i, trainingNum, averages[params->numMetrics] / i);
			for (int j = 0; j < params->numMetrics; j++) {
				printf("%s:%f  ", params->metrics[j]->toString().c_str(), averages[j] / i);
			}
			fit(&X[i], &y[i], averages, params);
			applyGradients(params);
		}
		printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, params->numEpochs, trainingNum, trainingNum, averages[params->numMetrics] / trainingNum);
		for (int j = 0; j < params->numMetrics; j++) {
			printf("%s:%f  ", params->metrics[j]->toString().c_str(), averages[j] / trainingNum);
		}
		int valNum = numData * params->valSplit;
		if (valNum > 0) {
			inputLayer->setBatchSize(valNum);
			predict(&X[trainingNum]);
			printf("ValLoss:%f  ", lossFunction->loss(outputLayer, &y[trainingNum]) / valNum);
			for (int i = 0; i < params->numMetrics; i++) {
				printf("Val%s:%f  ", params->metrics[i]->toString().c_str(), params->metrics[i]->loss(outputLayer, &y[trainingNum]) / valNum);
			}
		}
		printf("\n");
	}
}

void NeuralNetwork::test(int numData, double** X, double** y, int numMetrics, Loss** metrics) {
	inputLayer->setBatchSize(numData);
	predict(X);
	printf("TestLoss:%f  ", lossFunction->loss(outputLayer, y) / numData);
	for (int j = 0; j < numMetrics; j++) {
		printf("Test%s:%f  ", metrics[j]->toString().c_str(), metrics[j]->loss(outputLayer, y) / numData);
	}
	printf("\n");
}

void NeuralNetwork::setTrainable(bool trainable) {
	inputLayer->setTrainable(trainable);
}

void NeuralNetwork::save(string fileName) {
	ofstream file(fileName.c_str());
	int index = 0;
	for (int i = 0; i < Loss::NUM_LOSSES; i++) {
		if (lossFunction == Loss::ALL_LOSSES[i]) {
			index = i;
		}
	}
	file << index << ",";
	inputLayer->save(file);
	file.close();
}