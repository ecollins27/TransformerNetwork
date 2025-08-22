#include "Model1D.h"
#include <format>

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

void Model1D::updateAverages(Loss1D* lossFunction, float** y, atomic<float>* averages, int numMetrics, Loss1D** metrics) {
	for (int i = 0; i < numMetrics; i++) {
		averages[i].fetch_add(metrics[i]->loss(outputLayer, y));
	}
	averages[numMetrics].fetch_add(lossFunction->loss(outputLayer, y));
}

void Model1D::evaluateValidation(OperationQueue* predict, Loss1D* lossFunction, int valNum, Dataset* valData, int batchSize, int numMetrics, Loss1D** metrics, atomic<float>* averages, int threadID, barrier<>* sync) {
	if (threadID == 0) {
		for (int i = 0; i < numMetrics + 1; i++) {
			averages[i].store(0);
		}
	}
	for (int i = 0; i < valNum; i += batchSize) {
		if (threadID == 1) {
			if (valData->sparseX) {
				inputLayer->setSparseInput((int*)&valData->X[i]);
			}
			else {
				inputLayer->setInput((float**)&valData->X[i]);
			}
		}
		sync->arrive_and_wait();
		predict->threadRun(threadID);
		sync->arrive_and_wait();
		if (threadID == 0) {
			updateAverages(lossFunction, (float**)&valData->y[i], averages, numMetrics, metrics);
		}
	}
	sync->arrive_and_wait();
	if (threadID == 0) {
		printf("ValData  Loss:%f  ", averages[numMetrics].load() / valNum);
		for (int j = 0; j < numMetrics; j++) {
			printf("%s:%f  ", metrics[j]->toString().c_str(), averages[j].load() / valNum);
		}
	}
}

void Model1D::formatData(Dataset*& trainingData, Dataset*& valData, int& trainingNum, int& valNum, float valSplit, bool useSplitVal, int batchSize) {
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
	printf("TrainingNum: %d\n", trainingNum);
	printf("ValNum: %d\n", valNum);
}

void Model1D::threadFit(int trainingNum, Dataset* trainingData, int valNum, Dataset* valData, Loss1D* lossFunction, OperationQueue* forwardProp, OperationQueue* backProp, OperationQueue* applyGradients, OperationQueue* predict, int numMetrics, Loss1D** metrics, string header, atomic<float>* averages, float learningRate, int batchSize, int threadID, barrier<>* sync) {
	if (threadID == 1) {
		for (int i = 0; i < numMetrics + 1; i++) {
			averages[i].store(0);
		}
	}
	for (int i = 0; i < trainingNum; i += batchSize) {
		if (threadID == 0) {
			t++;
			if (trainingData->sparseX) {
				inputLayer->setSparseInput((int*)&trainingData->X[i]);
			}
			else {
				inputLayer->setInput((float**)&trainingData->X[i]);
			}
		}
		else if (threadID == 1) {
			printf("\r%s  %d/%d  Loss:%f  ", header.c_str(), i, trainingData, averages[numMetrics].load() / i);
			for (int j = 0; j < numMetrics; j++) {
				printf("%s:%f  ", metrics[j]->toString().c_str(), averages[j].load() / i);
			}
		}
		sync->arrive_and_wait();
		forwardProp->threadRun(threadID);
		sync->arrive_and_wait();
		if (threadID == 0) {
			lossFunction->differentiate(outputLayer, (float**)&trainingData->y[i]);
		}
		else if (threadID == 1) {
			updateAverages(lossFunction, (float**)&trainingData->y[i], averages, numMetrics, metrics);
		}
		sync->arrive_and_wait();
		backProp->threadRun(threadID);
		sync->arrive_and_wait();
		applyGradients->threadRun(threadID);
	}
	sync->arrive_and_wait();
	evaluateValidation(predict, lossFunction, valNum, valData, batchSize, numMetrics, metrics, averages, threadID, sync);
}

void Model1D::fit(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics, TrainingParams* params) {
	Dataset* trainingData = data;
	Dataset* valData;

	float valSplit = params->get<TrainingParams::VAL_SPLIT, float>();
	int batchSize = params->get<TrainingParams::BATCH_SIZE, int>();
	int numEpochs = params->get<TrainingParams::NUM_EPOCHS, int>();
	float learningRate = params->get<TrainingParams::LEARNING_RATE, float>();
	valData = params->get<TrainingParams::VAL_DATA, Dataset*>();
	inputLayer->setOptimizer(params->get<TrainingParams::OPTIMIZER, Optimizer<>*>());
	bool useSplitVal = valData == NULL;
	int trainingNum, valNum;
	formatData(trainingData, valData, trainingNum, valNum, valSplit, useSplitVal, batchSize);
	inputLayer->setBatchSize(batchSize);
	atomic<float>* averages = new atomic<float>[numMetrics + 1];

	int numThreads = 3;
	barrier<> sync(numThreads);
	thread* threads = new thread[numThreads];
	OperationQueue forwardProp(numThreads);
	OperationQueue predict(numThreads);
	OperationQueue backProp(numThreads);
	OperationQueue applyGradients(numThreads);
	inputLayer->initForwardPropQueue(forwardProp);
	inputLayer->initPredictQueue(predict);
	outputLayer->initBackPropQueue(backProp);
	inputLayer->initApplicationQueue(applyGradients, learningRate, t);
	forwardProp.finalize();
	predict.finalize();
	backProp.finalize();
	applyGradients.finalize();
	for (int epoch = 0; epoch < numEpochs; epoch++) {
		trainingData->shuffle();
		if (useSplitVal) {
			valData = trainingData->getMiniBatch(trainingNum, valNum);
		}
		else {
			valData->shuffle();
		}
		forwardProp.reset();
		predict.reset();
		backProp.reset();
		applyGradients.reset();
		 //int trainingNum, Dataset* trainingData, int valNum, Dataset* valData, Loss1D* lossFunction, OperationQueue* forwardProp, OperationQueue* backProp, OperationQueue* applyGradients,
		 // OperationQueue* predict, int numMetrics, Loss1D** metrics, string header, atomic<float>* averages, float learningRate, int batchSize, int& t, int threadID, barrier<>* sync
		for (int i = 0; i < numThreads; i++) {
			threads[i] = thread(&Model1D::threadFit, this, trainingNum, trainingData, valNum, valData, lossFunction, &forwardProp, &backProp, &applyGradients, &predict, numMetrics, metrics, 
				format("Epoch {}/{}", epoch, numEpochs), averages, learningRate, batchSize, i, &sync);
		}
		for (int i = 0; i < numThreads; i++) {
			threads[i].join();
		}
		printf("\n");
	}
}

void Model1D::test(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics) {
}

void Model1D::save(string filename) {
	ofstream file(filename.c_str());
	file << MODEL_NAME << "\n";
	inputLayer->save(file);
	file.close();
}

void Model1D::printLayers() {
	Layer* layer = inputLayer;
	while (layer->nextLayer != NULL) {
		printf("%s\n", typeid(*layer).name());
		layer = layer->nextLayer;
	}
}

int Model1D::getNumParameters() {
	return inputLayer->getNumParameters();
}