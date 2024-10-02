#include "NeuralNetwork.h"
#include "BatchNormalization.h"

NeuralNetwork::NeuralNetwork(int inputSize) {
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
	double valSplit = *((double*)(params->get(TrainingParams::VAL_SPLIT)));
	int batchSize = *((int*)(params->get(TrainingParams::BATCH_SIZE)));
	int numEpochs = *((int*)(params->get(TrainingParams::NUM_EPOCHS)));
	double learningRate = *((double*)(params->get(TrainingParams::LEARNING_RATE)));
	inputLayer->setOptimizer((Optimizer*)params->get(TrainingParams::OPTIMIZER));
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
		for (int i = 0; i < trainingNum - batchSize; i += batchSize) {
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
		int valNum = numData * valSplit;
		if (valNum > 0) {
			inputLayer->setBatchSize(valNum);
			predict(&X[trainingNum]);
			printf("ValLoss:%f  ", lossFunction->loss(outputLayer, &y[trainingNum]) / valNum);
			for (int i = 0; i < numMetrics; i++) {
				printf("Val%s:%f  ", metrics[i]->toString().c_str(), metrics[i]->loss(outputLayer, &y[trainingNum]) / valNum);
			}
		}
		printf("\n");
	}
}

void NeuralNetwork::test(Loss* lossFunction, int numData, double** X, double** y, int numMetrics, Loss** metrics) {
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
	inputLayer->save(file);
	file.close();
}