#include "Model1D.h"

Model1D::Model1D(int inputSize) {
	inputLayer = { new Input1D(inputSize) };
	outputLayer = inputLayer;
	t = 0;
}

void Model1D::addLayer(Layer1D* layer) {
	layer->setPrevLayer(outputLayer);
	outputLayer->setNextLayer(layer);
	outputLayer = layer;
}

void Model1D::applyGradients(float learningRate) {
	t++;
	inputLayer->applyGradients(learningRate, t);
}

int Model1D::getNumParameters() {
	return inputLayer->getNumParameters();
}

void Model1D::predict(float** input) {
	inputLayer->setInput(input);
	inputLayer->predict(0);
}

void Model1D::forwardPropagate(float** input) {
	inputLayer->setInput(input);
	inputLayer->forwardPropagate(0);
}

void Model1D::backPropagate(Loss* lossFunction, float** yTrue) {
	lossFunction->differentiate(outputLayer, yTrue, 0, true);
	outputLayer->backPropagate(0);
}

void Model1D::fit(Loss* lossFunction, float** X, float** y, float* losses, int numMetrics, Loss** metrics, TrainingParams* params) {
	forwardPropagate(X);
	backPropagate(lossFunction, y);
	for (int i = 0; i < numMetrics; i++) {
		losses[i] += metrics[i]->loss(outputLayer, y, 0, true);
	}
	losses[numMetrics] += lossFunction->loss(outputLayer, y, 0, true);
}

void Model1D::shuffle(int numData, float** X, float** y) {
	for (int i = 0; i < numData; i++) {
		int index = (int)(numData * ((float)rand() / (RAND_MAX + 1)));
		swap(X[i], X[index]);
		swap(y[i], y[index]);
	}
}

void Model1D::fit(Loss* lossFunction, int numData, float** X, float** y, int numMetrics, Loss** metrics, TrainingParams* params) {
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
		for (int i = 0; i < trainingNum; i += batchSize) {
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
				averages[numMetrics] += lossFunction->loss(outputLayer, &y[i], 0, true);
				for (int j = 0; j < numMetrics; j++) {
					averages[j] += metrics[j]->loss(outputLayer, &y[i], 0, true);
				}
			}
			int remaining = numData - i;
			if (remaining > 0) {
				inputLayer->setBatchSize(remaining);
				predict(&X[i]);
				averages[numMetrics] += lossFunction->loss(outputLayer, &y[i], 0, true);
				for (int j = 0; j < numMetrics; j++) {
					averages[j] += metrics[j]->loss(outputLayer, &y[i], 0, true);
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

void Model1D::test(Loss* lossFunction, int numData, float** X, float** y, int numMetrics, Loss** metrics) {
	inputLayer->setBatchSize(16);
	float* averages = new float[numMetrics + 1];
	for (int i = 0; i <= numData - 16; i += 16) {
		predict(&X[i]);
		averages[numMetrics] += lossFunction->loss(outputLayer, &y[i], 0, true);
		for (int j = 0; j < numMetrics; j++) {
			averages[j] += metrics[j]->loss(outputLayer, &y[i], 0, true);
		}
	}
	if (numData % 16 > 0) {
		inputLayer->setBatchSize(numData % 16);
		predict(&X[numData - numData % 16]);
		averages[numMetrics] += lossFunction->loss(outputLayer, &y[numData - numData % 16], 0, true);
		for (int j = 0; j < numMetrics; j++) {
			averages[j] += metrics[j]->loss(outputLayer, &y[numData - numData % 16], 0, true);
		}
	}
	printf("TestLoss:%f  ", averages[numMetrics] / numData);
	for (int j = 0; j < numMetrics; j++) {
		printf("Test%s:%f  ", metrics[j]->toString().c_str(), averages[j] / numData);
	}
	printf("\n");
}

void Model1D::save(string filename) {
	ofstream file(filename.c_str());
	file << "NeuralNetwork\n";
	inputLayer->save(file);
	file.close();
}