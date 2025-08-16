#include "Model2DTo1D.h"
#include "ResidualSave2D.h"
#include "ResidualAdd2D.h"
#include "LayerNormalization2D.h"
#include "Dense2D.h"
#include "Gated2D.h"
#include "TransformerAttention.h"
#include <format>

const string Model2DTo1D::MODEL_NAME = "Model2DTo1D";
int Model2DTo1D::NUM_CORES = 12;

Model2DTo1D::Model2DTo1D(int inputSize) {
	inputLayer = new Input2D(inputSize);
	tempLayer = inputLayer;
	outputLayer = NULL;
	t = 0;
	forwardThreadCount = { -1 };
	progress = { 0 };
}

void Model2DTo1D::addLayer(Layer* layer) {
	if (outputLayer == NULL) {
		layer->setPrevLayer(tempLayer);
		tempLayer->setNextLayer(layer);
		if (Layer::instanceOf<Layer1D>(layer)) {
			outputLayer = (Layer1D*)layer;
		}
		else {
			tempLayer = layer;
		}
	}
	else {
		layer->setPrevLayer(outputLayer);
		outputLayer->setNextLayer(layer);
		outputLayer = (Layer1D*)layer;
	}
}

Layer* Model2DTo1D::getLayer(int index) {
	Layer* layer = inputLayer;
	for (int i = 0; i < index; i++) {
		layer = layer->nextLayer;
	}
	return layer;
}

void Model2DTo1D::updateAverages(Loss1D* lossFunction, float** y, atomic<float>* averages, int numMetrics, Loss1D** metrics) {
	for (int i = 0; i < numMetrics; i++) {
		averages[i].fetch_add(metrics[i]->loss(outputLayer, y));
	}
	averages[numMetrics].fetch_add(lossFunction->loss(outputLayer, y));
}

int Model2DTo1D::getNumParameters() {
	return inputLayer->getNumParameters();
}

void Model2DTo1D::evaluateValidation(OperationQueue* predict, Loss1D* lossFunction, int valNum, Dataset* valData, int batchSize, int numMetrics, Loss1D** metrics, atomic<float>* averages, int threadID, barrier<>* sync) {
	if (threadID == 0) {
		for (int i = 0; i < numMetrics + 1; i++) {
			averages[i].store(0);
		}
	}
	for (int i = 0; i < valNum; i += batchSize) {
		if (threadID == 1) {
			t++;
			inputLayer->setNumTokens(&valData->numTokens[i]);
			if (valData->sparseX) {
				inputLayer->setSparseInput((int**)&valData->X[i]);
			}
			else {
				inputLayer->setInput((float***)&valData->X[i]);
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

void Model2DTo1D::formatData(Dataset*& trainingData, Dataset*& valData, int& trainingNum, int& valNum, float valSplit, bool useSplitVal, int batchSize) {
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

void Model2DTo1D::threadFit(int trainingNum, Dataset* trainingData, int valNum, Dataset* valData, Loss1D* lossFunction, OperationQueue* forwardProp, OperationQueue* backProp, OperationQueue* applyGradients, OperationQueue* predict, int numMetrics, Loss1D** metrics, string header, atomic<float>* averages, float learningRate, int batchSize, int threadID, barrier<>* sync) {
	if (threadID == 0) {
		for (int i = 0; i < numMetrics + 1; i++) {
			averages[i].store(0);
		}
	}
	for (int i = 0; i < trainingNum; i += batchSize) {
		if (threadID == 0) {
			t++;
			inputLayer->setNumTokens(&trainingData->numTokens[i]);
			if (trainingData->sparseX) {
				inputLayer->setSparseInput((int**)&trainingData->X[i]);
			}
			else {
				inputLayer->setInput((float***)&trainingData->X[i]);
			}
		} else if (threadID == 1) {
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

void Model2DTo1D::addTransformer(int numHeads, int keySize, int valueSize) {
	int size = tempLayer->size;
	ResidualSave2D* rs1 = { new ResidualSave2D() };
	this->addLayer(rs1);
	this->addLayer({ new TransformerAttention(numHeads, keySize, valueSize) });
	this->addLayer({ new ResidualAdd2D(rs1) });
	this->addLayer({ new LayerNormalization2D() });
	ResidualSave2D* rs2 = { new ResidualSave2D() };
	this->addLayer(rs2);
	this->addLayer({ new Dense2D(Activation::SWISH, size) });
	this->addLayer({ new Dense2D(Activation::SWISH, size) });
	this->addLayer({ new ResidualAdd2D(rs2) });
	this->addLayer({ new LayerNormalization2D() });
}

void Model2DTo1D::formatData(Dataset*& data, Dataset*& trainingData, Dataset*& valData, int& trainingNum, int& valNum, int& maxTokenSize, int& valMaxTokenSize, float valSplit, bool useSplitVal, int batchSize) {
	maxTokenSize = data->getMaxNumTokens();
	if (maxTokenSize > MAX_NUM_TOKENS) {
		trainingData = partitionData(data);
		maxTokenSize = MAX_NUM_TOKENS;
	}
	else {
		trainingData = data;
	}
	if (useSplitVal) {
		trainingNum = (int)(trainingData->numData * (1 - valSplit));
		valNum = trainingData->numData - trainingNum;
	}
	else {
		int valMaxTokenSize = valData->getMaxNumTokens();
		if (valMaxTokenSize > MAX_NUM_TOKENS) {
			valData = partitionData(valData);
			maxTokenSize = MAX_NUM_TOKENS;
		}
		else if (valMaxTokenSize > maxTokenSize) {
			maxTokenSize = valMaxTokenSize;
		}
		trainingNum = trainingData->numData;
		valNum = valData->numData;
	}
	trainingNum -= trainingNum % batchSize;
	valNum -= valNum % batchSize;
}

void Model2DTo1D::fit(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics, TrainingParams* params) {
	if (outputLayer == NULL) {
		throw invalid_argument("Model2DTo1D must have 1D output");
	}
	Dataset* trainingData;
	Dataset* valData;

	float valSplit = params->get<TrainingParams::VAL_SPLIT, float>();
	int batchSize = params->get<TrainingParams::BATCH_SIZE, int>();
	int numEpochs = params->get<TrainingParams::NUM_EPOCHS, int>();
	float learningRate = params->get<TrainingParams::LEARNING_RATE, float>();
	valData = params->get<TrainingParams::VAL_DATA, Dataset*>();
	inputLayer->setOptimizer(params->get<TrainingParams::OPTIMIZER, Optimizer<>*>());

	int maxTokenSize, valMaxTokenSize;
	int trainingNum, valNum;
	bool useSplitVal = valData == NULL;

	formatData(data, trainingData, valData, trainingNum, valNum, maxTokenSize, valMaxTokenSize, valSplit, useSplitVal, batchSize);

	inputLayer->setMaxNumTokens(maxTokenSize);
	inputLayer->setBatchSize(batchSize);
	float* averages = new float[numMetrics + 1];
	string timeEstimation = "";
	int numThreads = 7;
	barrier<> sync(numThreads);
	thread* threads = new thread[numThreads];
	OperationQueue forwardProp(numThreads);
	OperationQueue predict(numThreads);
	OperationQueue backProp(numThreads);
	OperationQueue applyGradients(numThreads);
	inputLayer->initForwardPropQueue(forwardProp);
	inputLayer->initPredictQueue(predict);
	inputLayer->initBackPropQueue(backProp);
	inputLayer->initApplicationQueue(applyGradients, learningRate, t);

	auto start = high_resolution_clock::now();
	for (int epoch = 0; epoch < numEpochs; epoch++) {
		trainingData->shuffle();
		if (useSplitVal) {
			valData = trainingData->getMiniBatch(trainingNum, valNum);
		}
		else {
			valData->shuffle();
		}
		 //int trainingNum, Dataset* trainingData, int valNum, Dataset* valData, Loss1D* lossFunction, OperationQueue* forwardProp, OperationQueue* backProp, OperationQueue* applyGradients,
		 //OperationQueue* predict, int numMetrics, Loss1D** metrics, string header, atomic<float>* averages, float learningRate, int batchSize, int& t, int threadID, barrier<>* sync
		//for (int i = 0; i < numThreads; i++) {
		//	threads[i] = thread(&Model2DTo1D::threadFit, this, trainingNum, trainingData, valNum, valData, lossFunction, &forwardProp, &backProp, &applyGradients, &predict, numMetrics, metrics,
		//		format("Epoch {}/{}", epoch, numEpochs), averages, learningRate, batchSize, i, &sync);
		//}
		//for (int i = 0; i < numThreads; i++) {
		//	threads[i].join();
		//}
		printf("\n");
	}
}

void Model2DTo1D::test(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics) {
}

string Model2DTo1D::estimateTime(auto start, float progress) {
	auto current = high_resolution_clock::now();
	auto duration = duration_cast<seconds>(current - start);
	int seconds = duration.count();
	seconds = (int)(seconds / progress);
	string time = "";
	if (seconds >= 86400) {
		time += to_string(seconds / 86400) + " days ";
		seconds = seconds % 86400;
	} if (seconds >= 3600) {
		time += to_string(seconds / 3600) + " hours ";
		seconds = seconds % 3600;
	} if (seconds >= 60) {
		time += to_string(seconds / 60) + " minutes ";
		seconds = seconds % 60;
	} if (seconds > 0) {
		time += to_string(seconds) + " seconds";
	}
	return time;
}

Dataset* Model2DTo1D::partitionData(Dataset* data) {
	int numData = data->numData;
	for (int i = 0; i < data->numData; i++) {
		if (data->numTokens[i] > MAX_NUM_TOKENS) {
			numData += data->numTokens[i] - MAX_NUM_TOKENS;
		}
	}
	void** newX = new void* [numData];
	void** newY = new void* [numData];
	int* numTokens = new int[numData];
	int counter = 0;
	for (int i = 0; i < data->numData; i++) {
		numTokens[counter] = data->numTokens[i] < MAX_NUM_TOKENS ? data->numTokens[i] : MAX_NUM_TOKENS;
		if (data->sparseX) {
			newX[counter] = &((int**)data->X)[i][0];
		}
		else {
			newX[counter] = &((float***)data->X)[i][0];
		}
		newY[counter] = data->y[i];
		counter++;
		for (int j = 1; j <= data->numTokens[i] - MAX_NUM_TOKENS; j++) {
			numTokens[counter] = MAX_NUM_TOKENS;
			if (data->sparseX) {
				newX[counter] = &((int**)data->X)[i][j];
			}
			else {
				newX[counter] = &((float***)data->X)[i][j];
			}
			newY[counter] = data->y[i];
			counter++;
		}
	}
	return new Dataset(numData, numTokens, newX, newY, data->sparseX);
}

void Model2DTo1D::save(string filename) {
	ofstream file(filename.c_str());
	file << MODEL_NAME << "\n";
	inputLayer->save(file);
	file.close();
}

void Model2DTo1D::printLayers() {
	Layer* layer = inputLayer;
	while (layer->nextLayer != NULL) {
		printf("%s\n", typeid(*layer).name());
		layer = layer->nextLayer;
	}
}