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

void Model1D::updateAverages(Loss1D* lossFunction, float** y, float* averages, int numMetrics, Loss1D** metrics) {
	for (int i = 0; i < numMetrics; i++) {
		averages[i] += metrics[i]->loss(outputLayer, y);
	}
	averages[numMetrics] += lossFunction->loss(outputLayer, y);
}

void Model1D::evaluateValidation(Loss1D* lossFunction, int numData, float** X, float** y, int batchSize, int numMetrics, Loss1D** metrics) {
	float* averages = new float[numMetrics + 1];
	for (int i = 0; i < numMetrics + 1; i++) {
		averages[i] = 0;
	}
	for (int i = 0; i < numData; i += batchSize) {
		predict(&X[i]);
		updateAverages(lossFunction, &y[i], averages, numMetrics, metrics);
	}
	printf("  ValLoss:%f  ", averages[numMetrics] / numData);
	for (int j = 0; j < numMetrics; j++) {
		printf("Val%s:%f  ", metrics[j]->toString().c_str(), averages[j] / numData);
	}
}

void Model1D::backPropagate(Loss1D* lossFunction, float** yTrue) {
	lossFunction->differentiate(outputLayer, yTrue);
	outputLayer->backPropagate(0);
}

void Model1D::shuffle(int numData, float** X, float** y) {
	for (int i = 0; i < numData; i++) {
		int index = (int)(numData * ((float)rand() / (RAND_MAX + 1)));
		swap(X[i], X[index]);
		swap(y[i], y[index]);
	}
}

void Model1D::fit(Loss1D* lossFunction, int numData, float** X, float** y, int numMetrics, Loss1D** metrics, TrainingParams* params) {
	float valSplit = params->get<TrainingParams::VAL_SPLIT, float>();
	int batchSize = params->get<TrainingParams::BATCH_SIZE, int>();
	int numEpochs = params->get<TrainingParams::NUM_EPOCHS, int>();
	float learningRate = params->get<TrainingParams::LEARNING_RATE, float>();
	float** XVal = params->get<TrainingParams::X_VAL, float**>();
	float** yVal = params->get<TrainingParams::Y_VAL, float**>();
	bool useSplitVal = XVal == NULL;
	inputLayer->setOptimizer(params->get<TrainingParams::OPTIMIZER, Optimizer*>());
	float* averages = new float[numMetrics + 1];
	int trainingNum = useSplitVal ? ((int)(numData * (1 - valSplit))) : numData;
	trainingNum -= trainingNum % batchSize;
	int valNum = useSplitVal ? (numData - trainingNum) : params->get<TrainingParams::VAL_SIZE, int>();
	valNum -= valNum % batchSize;
	inputLayer->setBatchSize(batchSize);
	for (int epoch = 0; epoch < numEpochs; epoch++) {
		shuffle(numData, X, y);
		for (int i = 0; i < numMetrics + 1; i++) {
			averages[i] = 0;
		}
		for (int i = 0; i < trainingNum; i += batchSize) {
			printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, numEpochs, i, trainingNum, averages[numMetrics] / i);
			for (int j = 0; j < numMetrics; j++) {
				printf("%s:%f  ", metrics[j]->toString().c_str(), averages[j] / i);
			}
			forwardPropagate(&X[i]);
			backPropagate(lossFunction, &y[i]);
			updateAverages(lossFunction, &y[i], averages, numMetrics, metrics);
			applyGradients(learningRate);
		}
		printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, numEpochs, trainingNum, trainingNum, averages[numMetrics] / trainingNum);
		for (int j = 0; j < numMetrics; j++) {
			printf("%s:%f  ", metrics[j]->toString().c_str(), averages[j] / trainingNum);
		}
		if (valNum > 0) {
			if (useSplitVal) {
				evaluateValidation(lossFunction, valNum, &X[trainingNum], &y[trainingNum], batchSize, numMetrics, metrics);
			}
			else {
				evaluateValidation(lossFunction, valNum, XVal, yVal, batchSize, numMetrics, metrics);
			}
		}
		printf("\n");
	}
}

void Model1D::test(Loss1D* lossFunction, int numData, float** X, float** y, int numMetrics, Loss1D** metrics) {
	inputLayer->setBatchSize(1);
	float* averages = new float[numMetrics + 1];
	for (int i = 0; i < numData; i++) {
		predict(&X[i]);
		updateAverages(lossFunction, &y[i], averages, numMetrics, metrics);
	}
	printf("TestLoss:%f  ", averages[numMetrics] / numData);
	for (int j = 0; j < numMetrics; j++) {
		printf("Test%s:%f  ", metrics[j]->toString().c_str(), averages[j] / numData);
	}
	printf("\n");
}

void Model1D::save(string filename) {
	ofstream file(filename.c_str());
	file << "Model1D\n";
	inputLayer->save(file);
	file.close();
}