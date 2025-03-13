#include "Model2DTo1D.h"
#include "ResidualSave2D.h"
#include "ResidualAdd2D.h"
#include "LayerNormalization2D.h"
#include "Dense2D.h"
#include "Gated2D.h"
#include "MultiHeadAttention.h"

int Model2DTo1D::NUM_CORES = 12;

Model2DTo1D::Model2DTo1D(int inputSize) {
	inputLayer = new Input2D(inputSize);
	tempLayer = inputLayer;
	t = 0;
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

void Model2DTo1D::applyGradients(float learningRate) {
	t++;
	inputLayer->applyGradients(learningRate, t);
}

void Model2DTo1D::updateAverages(Loss1D* lossFunction, float** y, float* averages, int numMetrics, Loss1D** metrics) {
	for (int j = 0; j < numMetrics; j++) {
		averages[j] += metrics[j]->loss(outputLayer, y);
	}
	averages[numMetrics] += lossFunction->loss(outputLayer, y);
}

int Model2DTo1D::getNumParameters() {
	return inputLayer->getNumParameters();
}

void Model2DTo1D::predict(void* input, bool sparse, int thread) {
	if (sparse) {
		inputLayer->setSparseInput(thread, (int*)input);
	}
	else {
		inputLayer->setInput(thread, (float**)input);
	}
	inputLayer->predict(thread);
}

void Model2DTo1D::forwardPropagate(void* input, bool sparse, int thread) {
	if (sparse) {
		inputLayer->setSparseInput(thread, (int*)input);
	} else {
		inputLayer->setInput(thread, (float**)input);
	}
	inputLayer->forwardPropagate(thread);
}

void Model2DTo1D::backPropagate(Loss1D* lossFunction, int thread) {
	outputLayer->backPropagate(thread);
}

void Model2DTo1D::evaluateValidation(Loss1D* lossFunction, Dataset* valData, int batchSize, int numMetrics, Loss1D** metrics) {
	int valSize = valData->numData;
	float* averages = new float[numMetrics + 1];
	for (int i = 0; i < numMetrics + 1; i++) {
		averages[i] = 0;
	}
	thread* threads = new thread[batchSize];
	for (int i = 0; i < valSize; i += batchSize) {
		inputLayer->setNumTokens(&valData->numTokens[i]);
		for (int k = 0; k < batchSize; k++) {
			threads[k] = thread(&Model2DTo1D::predict, this, valData->X[i + k],valData->sparseX, k);
		}
		for (int k = 0; k < batchSize; k++) {
			threads[k].join();
		}
		updateAverages(lossFunction, (float**)(&valData->y[i]), averages, numMetrics, metrics);
	}
	printf("  ValLoss:%f  ", averages[numMetrics] / valSize);
	for (int j = 0; j < numMetrics; j++) {
		printf("Val%s:%f  ", metrics[j]->toString().c_str(), averages[j] / valSize);
	}
}

void Model2DTo1D::addTransformer(int numHeads, int keySize, int valueSize) {
	int size = tempLayer->size;
	ResidualSave2D* rs1 = { new ResidualSave2D() };
	this->addLayer(rs1);
	this->addLayer({ new MultiHeadAttention(numHeads, keySize, valueSize) });
	this->addLayer({ new ResidualAdd2D(rs1) });
	this->addLayer({ new LayerNormalization2D() });
	ResidualSave2D* rs2 = { new ResidualSave2D() };
	this->addLayer(rs2);
	this->addLayer({ new Dense2D(Activation::SWISH, size) });
	this->addLayer({ new Dense2D(Activation::SWISH, size) });
	this->addLayer({ new ResidualAdd2D(rs2) });
	this->addLayer({ new LayerNormalization2D() });
}

void Model2DTo1D::addLinformer(int numHeads, int keySize, int valueSize, int projSize) {
	int size = tempLayer->size;
	ResidualSave2D* rs1 = { new ResidualSave2D() };
	this->addLayer(rs1);
	this->addLayer({ new LinformerAttention(numHeads, keySize, valueSize, projSize) });
	this->addLayer({ new ResidualAdd2D(rs1) });
	this->addLayer({ new LayerNormalization2D() });
	ResidualSave2D* rs2 = { new ResidualSave2D() };
	this->addLayer(rs2);
	this->addLayer({ new Dense2D(Activation::SWISH, size) });
	this->addLayer({ new Dense2D(Activation::SWISH, size) });
	this->addLayer({ new ResidualAdd2D(rs2) });
	this->addLayer({ new LayerNormalization2D() });
}

void Model2DTo1D::fit(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics, TrainingParams* params) {
	if (outputLayer == NULL) {
		throw invalid_argument("Model2DTo1D must have 1D output");
	}
	Dataset* trainingData;
	Dataset* valData;

	int maxTokenSize = data->getMaxNumTokens();
	if (maxTokenSize > MAX_NUM_TOKENS) {
		trainingData = partitionData(data);
		maxTokenSize = MAX_NUM_TOKENS;
	}
	else {
		trainingData = data;
	}
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

	printf("MaxTokenSize:%d\n", maxTokenSize);
	inputLayer->setMaxNumTokens(maxTokenSize);
	inputLayer->setBatchSize(batchSize);
	thread* threads = new thread[batchSize];
	float* averages = new float[numMetrics + 1];
	float minLoss = INT_MAX;
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
		printf("%p\n", trainingData->y[53]);
		for (int i = 0; i < trainingNum; i += batchSize) {
			printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, numEpochs, i, trainingNum, averages[numMetrics] / i);
			for (int j = 0; j < numMetrics; j++) {
				printf("%s:%f  ", metrics[j]->toString().c_str(), averages[j] / i);
			}
			inputLayer->setNumTokens(&trainingData->numTokens[i]);
			for (int k = 0; k < batchSize; k++) {
				threads[k] = thread(&Model2DTo1D::forwardPropagate, this, trainingData->X[i + k], trainingData->sparseX, k);
			}
			for (int k = 0; k < batchSize; k++) {
				threads[k].join();
			}
			lossFunction->differentiate(outputLayer, (float**)(&trainingData->y[i]));
			for (int k = 0; k < batchSize; k++) {
				threads[k] = thread(&Model2DTo1D::backPropagate, this, lossFunction, k);
			}
			for (int k = 0; k < batchSize; k++) {
				threads[k].join();
			}
			updateAverages(lossFunction, (float**)(&trainingData->y[i]), averages, numMetrics, metrics);
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

void Model2DTo1D::test(Loss1D* lossFunction, Dataset* data, int numMetrics, Loss1D** metrics) {
	return;
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
	printf("%d %d\n", counter, numData);
	return new Dataset(numData, numTokens, newX, newY, data->sparseX);
}

void Model2DTo1D::save(string filename) {
	ofstream file(filename.c_str());
	file << "Model2DTo1D\n";
	inputLayer->save(file);
	file.close();
}