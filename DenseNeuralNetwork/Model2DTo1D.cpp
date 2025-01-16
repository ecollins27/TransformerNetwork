#include "Model2DTo1D.h"
#include "ResidualSave2D.h"
#include "ResidualAdd2D.h"
#include "LayerNormalization2D.h"
#include "Dense2D.h"
#include "Gated2D.h"
#include "MultiHeadAttention.h"

int Model2DTo1D::NUM_CORES = 1;

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

void Model2DTo1D::predict(float** input, int thread) {
	inputLayer->setInput(thread, input);
	inputLayer->predict(thread);
}

void Model2DTo1D::forwardPropagate(float** input, int thread) {
	inputLayer->setInput(thread, input);
	inputLayer->forwardPropagate(thread);
}

void Model2DTo1D::backPropagate(Loss1D* lossFunction, float** yTrue, int thread) {
	outputLayer->backPropagate(thread);
}

void Model2DTo1D::evaluateValidation(Loss1D* lossFunction, int valSize, float*** XVal, float** yVal, int* numTokens, int batchSize, int numMetrics, Loss1D** metrics) {
	float* averages = new float[numMetrics + 1];
	for (int i = 0; i < numMetrics + 1; i++) {
		averages[i] = 0;
	}
	thread* threads = new thread[batchSize];
	for (int i = 0; i < valSize; i += batchSize) {
		inputLayer->setNumTokens(&numTokens[i]);
		for (int k = 0; k < batchSize; k++) {
			threads[k] = thread(&Model2DTo1D::predict, this, XVal[i + k], k);
		}
		for (int k = 0; k < batchSize; k++) {
			threads[k].join();
		}
		updateAverages(lossFunction, &yVal[i], averages, numMetrics, metrics);
	}
	printf("  ValLoss:%f  ", averages[numMetrics] / valSize);
	for (int j = 0; j < numMetrics; j++) {
		printf("Val%s:%f  ", metrics[j]->toString().c_str(), averages[j] / valSize);
	}
}

void Model2DTo1D::addTransformerBlock(int numHeads, int keySize, int valueSize) {
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

void Model2DTo1D::fit(Loss1D* lossFunction, int numData, int* numTokens, float*** X, float** y, int numMetrics, Loss1D** metrics, TrainingParams* params) {
	if (outputLayer == NULL) {
		throw invalid_argument("Model2DTo1D must have 1D output");
	}
	int maxTokenSize = 0;
	float minLoss = INT_MAX;
	for (int i = 0; i < numData; i++) {
		if (numTokens[i] > maxTokenSize) {
			maxTokenSize = numTokens[i];
		}
	}
	float valSplit = params->get<TrainingParams::VAL_SPLIT, float>();
	int batchSize = params->get<TrainingParams::BATCH_SIZE, int>();
	int numEpochs = params->get<TrainingParams::NUM_EPOCHS, int>();
	float learningRate = params->get<TrainingParams::LEARNING_RATE, float>();
	int* valNumTokens = params->get<TrainingParams::VAL_NUM_TOKENS, int*>();
	float*** XVal = params->get<TrainingParams::X_VAL, float***>();
	float** yVal = params->get<TrainingParams::Y_VAL, float**>();
	bool useSplitVal = XVal == NULL;
	inputLayer->setOptimizer(params->get<TrainingParams::OPTIMIZER, Optimizer*>());
	float* averages = new float[numMetrics + 1];
	int trainingNum = useSplitVal? ((int)(numData * (1 - valSplit))):numData;
	trainingNum -= trainingNum % batchSize;
	int valNum = useSplitVal ? (numData - trainingNum) : params->get<TrainingParams::VAL_SIZE, int>();
	valNum -= valNum % batchSize;
	if (!useSplitVal) {
		for (int i = 0; i < valNum; i++) {
			if (valNumTokens[i] > maxTokenSize) {
				maxTokenSize = valNumTokens[i];
			}
		}
	}
	printf("MaxTokenSize:%d\n", maxTokenSize);
	inputLayer->setMaxNumTokens(maxTokenSize);
	inputLayer->setBatchSize(batchSize);
	thread* threads = new thread[batchSize];
	for (int epoch = 0; epoch < numEpochs; epoch++) {
		shuffle(numData, numTokens, X, y);
		for (int i = 0; i < numMetrics + 1; i++) {
			averages[i] = 0;
		}
		for (int i = 0; i < trainingNum; i += batchSize) {
			printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, numEpochs, i, trainingNum, averages[numMetrics] / i);
			for (int j = 0; j < numMetrics; j++) {
				printf("%s:%f  ", metrics[j]->toString().c_str(), averages[j] / i);
			}
			inputLayer->setNumTokens(&numTokens[i]);
			for (int k = 0; k < batchSize; k++) {
				threads[k] = thread(&Model2DTo1D::forwardPropagate, this, X[i + k], k);
			}
			for (int k = 0; k < batchSize; k++) {
				threads[k].join();
			}
			lossFunction->differentiate(outputLayer, &y[i]);
			for (int k = 0; k < batchSize; k++) {
				threads[k] = thread(&Model2DTo1D::backPropagate, this, lossFunction, &y[i], k);
			}
			for (int k = 0; k < batchSize; k++) {
				threads[k].join();
			}
			updateAverages(lossFunction, &y[i], averages, numMetrics, metrics);
			applyGradients(learningRate);
		}
		printf("\rEpoch %d/%d  %d/%d  Loss:%f  ", epoch + 1, numEpochs, trainingNum, trainingNum, averages[numMetrics] / trainingNum);
		for (int j = 0; j < numMetrics; j++) {
			printf("%s:%f  ", metrics[j]->toString().c_str(), averages[j] / trainingNum);
		}
		if (valNum > 0) {
			if (useSplitVal) {
				evaluateValidation(lossFunction, valNum, &X[trainingNum], &y[trainingNum], &numTokens[trainingNum], batchSize, numMetrics, metrics);
			}
			else {
				evaluateValidation(lossFunction, valNum, XVal, yVal, valNumTokens, batchSize, numMetrics, metrics);
			}
		}
		printf("\n");
	}
}

void Model2DTo1D::test(Loss1D* lossFunction, int numData, int* numTokens, float*** X, float** y, int numMetrics, Loss1D** metrics) {
	int maxTokenSize = 0;
	for (int i = 0; i < numData; i++) {
		if (numTokens[i] > maxTokenSize) {
			maxTokenSize = numTokens[i];
		}
	}
	printf("MaxTokenSize:%d\n", maxTokenSize);
	inputLayer->setMaxNumTokens(maxTokenSize);
	inputLayer->setBatchSize(1);
	float* averages = new float[numMetrics + 1];
	for (int i = 0; i < numMetrics + 1; i++) {
		averages[i] = 0;
	}
	for (int i = 0; i < numData; i++) {
		inputLayer->setNumTokens(&numTokens[i]);
		predict(X[i], 0);
		averages[numMetrics] += lossFunction->loss(outputLayer, &y[i]);
		for (int j = 0; j < numMetrics; j++) {
			averages[j] += metrics[j]->loss(outputLayer, &y[i]);
		}
		printf("\rTest %d/%d  NumTokens:%d  TestLoss:%f  ", i + 1, numData, numTokens[i], averages[numMetrics] / (i + 1));
		for (int j = 0; j < numMetrics; j++) {
			printf("Test%s:%f  ", metrics[j]->toString().c_str(), averages[j] / (i + 1));
		}
	}
	printf("\n");
}

void Model2DTo1D::shuffle(int numData, int* numTokens, float*** X, float** y) {
	for (int i = 0; i < numData; i++) {
		int index = (int)(numData * ((float)rand() / (RAND_MAX + 1)));
		swap(X[i], X[index]);
		swap(y[i], y[index]);
		swap(numTokens[i], numTokens[index]);
	}
}

void Model2DTo1D::save(string filename) {
	ofstream file(filename.c_str());
	file << "Model2DTo1D\n";
	inputLayer->save(file);
	file.close();
}