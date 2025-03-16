#include "Model1D.h"

const string Model1D::MODEL_NAME = "Model1D";

Model1D::Model1D(int inputSize) {
	inputLayer = { new Input1D(inputSize) };
	outputLayer = inputLayer;
	t = 0;
}

void Model1D::addLayer(Layer* layer) {
	if (dynamic_cast<Layer1D*>(layer) == NULL) {
		throw invalid_argument("Layer must be 1D");
	}
	layer->setPrevLayer(outputLayer);
	outputLayer->setNextLayer(layer);
	outputLayer = (Layer1D*)layer;
}

Layer* Model1D::getLayer(int index) {
	Layer* layer = inputLayer;
	for (int i = 0; i < index; i++) {
		layer = layer->nextLayer;
	}
	return layer;
}

void Model1D::applyGradients(float learningRate) {
	t++;
	inputLayer->applyGradients(learningRate, t);
}

int Model1D::getNumParameters() {
	return inputLayer->getNumParameters();
}

void Model1D::predict(void* input, bool sparse) {
	if (sparse) {
		inputLayer->setSparseInput((int*)input);
	}
	else {
		inputLayer->setInput((float**)input);
	}
	inputLayer->predict(0);
}

void Model1D::forwardPropagate(void* input, bool sparse) {
	if (sparse) {
		inputLayer->setSparseInput((int*)input);
	}
	else {
		inputLayer->setInput((float**)input);
	}
	inputLayer->forwardPropagate(0);
}

void Model1D::updateAverages(Loss1D* lossFunction, float** y, float* averages, int numMetrics, Loss1D** metrics) {
	for (int i = 0; i < numMetrics; i++) {
		averages[i] += metrics[i]->loss(outputLayer, y);
	}
	averages[numMetrics] += lossFunction->loss(outputLayer, y);
}

void Model1D::evaluateValidation(Loss1D* lossFunction, Dataset* valData, int batchSize, int numMetrics, Loss1D** metrics) {
	int valSize = valData->numData;
	float* averages = new float[numMetrics + 1];
	for (int i = 0; i < numMetrics + 1; i++) {
		averages[i] = 0;
	}
	thread* threads = new thread[batchSize];
	for (int i = 0; i < valSize; i += batchSize) {
		predict(valData->X[i], valData->sparseX);
		updateAverages(lossFunction, (float**)(&valData->y[i]), averages, numMetrics, metrics);
	}
	printf("  ValLoss:%f  ", averages[numMetrics] / valSize);
	for (int j = 0; j < numMetrics; j++) {
		printf("Val%s:%f  ", metrics[j]->toString().c_str(), averages[j] / valSize);
	}
}

void Model1D::backPropagate(Loss1D* lossFunction, float** yTrue) {
	lossFunction->differentiate(outputLayer, yTrue);
	outputLayer->backPropagate(0);
}

void Model1D::fit(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics, TrainingParams* params) {
	Dataset* trainingData = data;
	Dataset* valData;

	float valSplit = params->get<TrainingParams::VAL_SPLIT, float>();
	int batchSize = params->get<TrainingParams::BATCH_SIZE, int>();
	int numEpochs = params->get<TrainingParams::NUM_EPOCHS, int>();
	float learningRate = params->get<TrainingParams::LEARNING_RATE, float>();
	valData = params->get<TrainingParams::VAL_DATA, Dataset*>();
	inputLayer->setOptimizer(params->get<TrainingParams::OPTIMIZER, Optimizer*>());

	int trainingNum, valNum;
	bool useSplitVal = valData == NULL;
	if (useSplitVal) {
		trainingNum = (int)(trainingData->numData * (1 - valSplit));
		valNum = trainingData->numData - trainingNum;
	}
	else {
		trainingNum = trainingData->numData;
		valNum = valData->numData;
	}
	trainingNum -= trainingNum % batchSize;
	valNum -= valNum % batchSize;
	inputLayer->setBatchSize(batchSize);
	float* averages = new float[numMetrics + 1];
	for (int epoch = 0; epoch < numEpochs; epoch++) {
		trainingData->shuffle();
		if (useSplitVal) {
			valData = trainingData->getMiniBatch(trainingNum, valNum);
		}
		else {
			valData->shuffle();
		}
		for (int i = 0; i < numMetrics + 1; i++) {
			averages[i] = 0;
		}
		for (int i = 0; i < trainingNum; i += batchSize) {
			printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, numEpochs, i, trainingNum, averages[numMetrics] / i);
			for (int j = 0; j < numMetrics; j++) {
				printf("%s:%f  ", metrics[j]->toString().c_str(), averages[j] / i);
			}
			forwardPropagate((float**) &trainingData->X[i], trainingData->sparseX);
			backPropagate(lossFunction, (float**) &trainingData->y[i]);
			updateAverages(lossFunction, (float**) &trainingData->y[i], averages, numMetrics, metrics);
			applyGradients(learningRate);
		}
		printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, numEpochs, trainingNum, trainingNum, averages[numMetrics] / trainingNum);
		for (int j = 0; j < numMetrics; j++) {
			printf("%s:%f  ", metrics[j]->toString().c_str(), averages[j] / trainingNum);
		}
		if (valNum > 0) {
			evaluateValidation(lossFunction, valData, batchSize, numMetrics, metrics);
		}
		printf("\n");
	}
}

void Model1D::test(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics) {
	return;
}

void Model1D::save(string filename) {
	ofstream file(filename.c_str());
	file << MODEL_NAME << "\n";
	inputLayer->save(file);
	file.close();
}