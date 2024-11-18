#include "NeuralNetwork.h"
#include "BatchNormalization.h"
#include "GatedLayer.h"

NeuralNetwork::NeuralNetwork(int inputSize) {
	inputLayer = { new InputLayer(inputSize) };
	outputLayer = inputLayer;
	t = 0;
}

void NeuralNetwork::predict(float** input) {
	inputLayer->setInput(input);
	inputLayer->predict();
}

void NeuralNetwork::forwardPropagate(float** input) {
	inputLayer->setInput(input);
	inputLayer->forwardPropagate();
}

void NeuralNetwork::backPropagate(Loss* lossFunction, float** yTrue) {
	lossFunction->differentiate(outputLayer, yTrue);
	outputLayer->backPropagate();
}

void NeuralNetwork::fit(Loss* lossFunction, float** X, float** y, float* losses, int numMetrics, Loss** metrics, TrainingParams* params) {
	forwardPropagate(X);
	backPropagate(lossFunction, y);
	for (int i = 0; i < numMetrics; i++) {
		losses[i] += metrics[i]->loss(outputLayer, y);
	}
	losses[numMetrics] += lossFunction->loss(outputLayer, y);
}

void NeuralNetwork::shuffle(int numData, float** X, float** y) {
	for (int i = 0; i < numData; i++) {
		int index = (int)(numData * ((float)rand() / (RAND_MAX + 1)));
		swap(X[i], X[index]);
		swap(y[i], y[index]);
	}
}

void NeuralNetwork::fit(Loss* lossFunction, int numData, float** X, float** y, int numMetrics, Loss** metrics, TrainingParams* params) {
	float valSplit = params->get<float>(TrainingParams::VAL_SPLIT);
	int batchSize = params->get<int>(TrainingParams::BATCH_SIZE);
	int numEpochs = params->get<int>(TrainingParams::NUM_EPOCHS);
	float learningRate = params->get<float>(TrainingParams::LEARNING_RATE);
	inputLayer->setOptimizer(params->get<Optimizer*>(TrainingParams::OPTIMIZER));
	float* averages = NULL;
	averages = new float[numMetrics + 1];
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

void NeuralNetwork::test(Loss* lossFunction, int numData, float** X, float** y, int numMetrics, Loss** metrics) {
	inputLayer->setBatchSize(16);
	float* averages = new float[numMetrics + 1];
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

void NeuralNetwork::save(string filename) {
	ofstream file(filename.c_str());
	file << "NeuralNetwork\n";
	inputLayer->save(file);
	file.close();
}